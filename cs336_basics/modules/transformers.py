import os
import math
import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch import Tensor
from jaxtyping import Float, Int

from collections.abc import Callable, Iterable
from typing import Optional, BinaryIO, IO, Union


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        # create a Matrix of numbers (the weights) that the model can update.
        init.trunc_normal_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W 
        # x has shape (Batch_Size, in_features)
        # self.W has shape (in_features, out_features)
        # results in (Batch_Size, out_features)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Use torch.empty to allocate memory, then overwrite it with initialization.
        # Shape is (V, d_model)
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        
        # Initialize weights using trunc_normal_ as requested
        init.trunc_normal_(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Perform lookup WITHOUT nn.functional.embedding.
        # In PyTorch, passing a tensor of indices to a larger tensor 
        # automatically performs a lookup/gather operation.
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.
        
        Args:
            d_model (int): Hidden dimension of the model.
            eps (float): Epsilon value for numerical stability.
            device (torch.device | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()
        self.eps = eps
        
        # The learnable parameter 'g_i' from your equation.
        # We initialize it to ones (identity) so the layer initially does not 
        # change the magnitude of the features.
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor.
        
        Args:
            x (torch.Tensor): Input shape (..., d_model)
            
        Returns:
            torch.Tensor: Normalized output of same shape.
        """
        # 1. Upcast input to float32 to prevent overflow when squaring
        x_fp32 = x.to(torch.float32)

        # 2. Calculate sum of squares / d_model (The Mean of Squares)
        # We compute along the last dimension (d_model). 
        # keepdim=True allows us to divide x by this value easily via broadcasting.
        squared_mean = x_fp32.pow(2).mean(dim=-1, keepdim=True)

        # 3. Calculate RMS (add epsilon then sqrt)
        rms = torch.sqrt(squared_mean + self.eps)

        # 4. Normalize the input
        normalized = x_fp32 / rms

        # 5. Downcast back to original dtype and scale by the learnable weight (gain)
        return normalized.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        
        # 1. Calculate d_ff: 8/3 * d_model, rounded to multiple of 64
        # We perform the multiplication first to avoid precision loss
        hidden_dim = int((8 * d_model) / 3)
        # Round up to nearest multiple of 64
        self.d_ff = ((hidden_dim + 63) // 64) * 64
        
        # 2. Define the three linear layers
        # W1: Gate projection (d_model -> d_ff)
        # W2: Down projection (d_ff -> d_model)
        # W3: Up projection   (d_model -> d_ff)
        
        # Note: We use your custom Linear class which stores weights as (in, out)
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq, d_model)
        
        # 1. Projections
        # w1(x) and w3(x) project from d_model to d_ff
        gate = self.w1(x) 
        up = self.w3(x)
        
        # 2. SwiGLU Activation: (SiLU(Gate) * Up)
        # You can use F.silu or x * torch.sigmoid(x)
        # We use F.silu here as it is standard, but torch.sigmoid works too.
        act = torch.nn.functional.silu(gate)
        
        # Element-wise multiplication
        h = act * up
        
        # 3. Down projection
        return self.w2(h)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # 1. Calculate Frequencies
        # The prompt specifies theta_i,k = i * Theta^(-(2k-2)/d)
        # Note: In 0-indexed Python, the k-th pair corresponds to indices 2k and 2k+1.
        # The exponent is -(2k)/d.
        
        # Create a sequence [0, 2, 4, ... d_k-2]
        # We use torch.arange(0, d_k, 2)
        exponent = torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k
        freqs = 1.0 / (theta ** exponent) # Shape: (d_k / 2)

        # 2. Create the grid of angles for all possible positions
        # positions: [0, 1, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        
        # Outer product: angles[i, k] = pos[i] * freq[k]
        # Shape: (max_seq_len, d_k / 2)
        angles = torch.outer(positions, freqs)

        # 3. Expand to match input shape (d_k) for easy broadcasting
        # We need cos(theta) and sin(theta) repeated for the pairs.
        # i.e., [theta_0, theta_0, theta_1, theta_1, ...]
        angles = torch.repeat_interleave(angles, 2, dim=-1)

        # 4. Cache cos and sin
        # Register as buffers so they are saved with state_dict but not updated by optimizer
        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., seq_len, d_k)
            token_positions: shape (..., seq_len) with integer indices
        """
        # 1. Look up the specific cos/sin for the given token positions.
        # token_positions can be arbitrary shape (Batch, Seq), so we use standard indexing/embedding lookup.
        # F.embedding handles the lookup nicely: [Batch, Seq] -> [Batch, Seq, d_k]
        cos = nn.functional.embedding(token_positions, self.cos_cached)
        sin = nn.functional.embedding(token_positions, self.sin_cached)

        # 2. Unsqueeze for Heads
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # 3. Apply rotation formula
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        x_evens = x_pairs[..., 0]
        x_odds  = x_pairs[..., 1]
        x_rotated_pairs = torch.stack((-x_odds, x_evens), dim=-1)
        x_rotated = x_rotated_pairs.flatten(start_dim=-2)

        # SAFETY CHANGE: Cast cos/sin to x.dtype
        # This prevents errors if x is BF16 but buffers are FP32
        return (x * cos.to(x.dtype)) + (x_rotated * sin.to(x.dtype))


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax of vector x in a numerically stable way.
    
    Args:
        x: Input tensor.
        dim: The dimension to normalize over.
        
    Returns:
        The softmax normalized tensor.
    """
    # 1. Numerical Stability: Subtract the max value along the specific dimension.
    # We use keepdim=True so we can broadcast the subtraction.
    # torch.max returns (values, indices), so we take .values
    max_val = x.max(dim=dim, keepdim=True).values
    
    # x_shifted has a max value of 0, so exp(x_shifted) will never overflow.
    x_shifted = x - max_val
    
    # 2. Exponentiate
    exp_x = torch.exp(x_shifted)
    
    # 3. Normalize
    # Sum along the same dimension, keepdim=True for broadcasting division.
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    return exp_x / sum_exp


def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute Scaled Dot-Product Attention.
    
    Args:
        q: Query tensor of shape (..., seq_len_q, d_k)
        k: Key tensor of shape (..., seq_len_k, d_k)
        v: Value tensor of shape (..., seq_len_v, d_v)
        mask: Boolean mask of shape (..., seq_len_q, seq_len_k). 
              True means attend, False means ignore.
    """
    # d_k = q.size(-1)

    # # 1. Calculate Scores: (Q @ K^T) / sqrt(d_k)
    # # We transpose the last two dimensions of K to match (..., d_k, seq_len_k)
    # # Resulting shape: (..., seq_len_q, seq_len_k)
    # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # # 2. Apply Masking
    # if mask is not None:
    #     # The prompt says: "add a -infinity in any entry of the mask matrix that is False"
    #     # masked_fill takes a boolean mask where True indicates elements to fill.
    #     # Since our mask has False for items to hide, we use ~mask (logical NOT).
    #     scores = scores.masked_fill(~mask, -float('inf'))

    # # 3. Softmax
    # # Apply softmax to the last dimension (the key dimension) to get probabilities.
    # # You can use the 'softmax' function you implemented in the previous step, 
    # # or F.softmax if that isn't available in this scope. 
    # # Assuming 'softmax' from the previous step is available:
    # attn_probs = softmax(scores, dim=-1) 

    # # 4. Weighted Sum: (Probabilities @ V)
    # # Shape: (..., seq_len_q, seq_len_k) @ (..., seq_len_v, d_v) -> (..., seq_len_q, d_v)
    # output = torch.matmul(attn_probs, v)


    # return output
    d_k = q.size(-1)

    # 1. Calculate Scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. Apply Masking
    if mask is not None:
        scores = scores.masked_fill(~mask, -float('inf'))

    # F.softmax which is fused and optimized for speed.
    attn_probs = F.softmax(scores, dim=-1) 

    # 4. Weighted Sum
    output = torch.matmul(attn_probs, v)

    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Determine head dimension: d_k = d_v = d_model / h
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
            
        self.head_dim = d_model // num_heads
        
        # 1. Define Projections
        # We use your custom Linear class (or standard nn.Linear if you prefer).
        # Assuming your custom Linear stores weights as (in, out).
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self, 
        x: torch.Tensor, 
        rope_module: nn.Module = None, 
        token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (Batch, Seq, d_model)
            rope_module: Optional RotaryPositionalEmbedding module.
            token_positions: Optional position indices for RoPE.
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Apply Projections
        # Shape: (Batch, Seq, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape for Multi-Head Attention
        # Split d_model into (num_heads, head_dim)
        # We reshape to (Batch, Seq, Num_Heads, Head_Dim)
        # Then transpose to (Batch, Num_Heads, Seq, Head_Dim) so heads are treated as batch dim
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Apply RoPE (if provided)
        # RoPE applies to Q and K
        if rope_module is not None and token_positions is not None:
            # RoPE handles broadcasting internally now via unsqueeze check
            q = rope_module(q, token_positions)
            k = rope_module(k, token_positions)

        # Causal Mask (Lower Triangular)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # 5. Scaled Dot-Product Attention
        # Output shape: (Batch, Num_Heads, Seq, Head_Dim)
        # Recombine: (Batch, Heads, Seq, Dim) -> (Batch, Seq, Heads, Dim) -> (Batch, Seq, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(batch_size, seq_len, self.d_model)
        
        return self.o_proj(attn_out)

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        max_seq_len: int,
        theta: float,
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # SwiGLU components manually instantiated to allow external d_ff control
        self.ffn_w1 = Linear(d_model, d_ff, device=device, dtype=dtype) # Gate
        self.ffn_w2 = Linear(d_ff, d_model, device=device, dtype=dtype) # Down
        self.ffn_w3 = Linear(d_model, d_ff, device=device, dtype=dtype) # Up
        
        d_head = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta, d_head, max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # Attention Sublayer
        resid = x
        x = self.ln1(x)
        x = self.attn(x, rope_module=self.rope, token_positions=positions)
        x = x + resid
        
        # FFN Sublayer (SwiGLU)
        resid = x
        x = self.ln2(x)
        
        gate = self.ffn_w1(x)
        up = self.ffn_w3(x)
        act = F.silu(gate)
        h = act * up
        x = self.ffn_w2(h)
        
        x = x + resid
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Token Embeddings
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # 2. Transformer Blocks
        # stored in a ModuleList so PyTorch can register parameters
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        # 3. Final RMSNorm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        # 4. Language Model Head (Output Projection)
        # Projects back from d_model to vocab_size
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_indices: shape (Batch, Seq)
        Returns:
            Logits: shape (Batch, Seq, Vocab_Size)
        """
        # 1. Embed tokens
        x = self.token_embeddings(in_indices)
        
        # 2. Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x)
            
        # 3. Final Normalization
        x = self.ln_final(x)
        
        # 4. Output Projection
        logits = self.lm_head(x)
        
        return logits


def compute_cross_entropy_loss(
    logits: Float[Tensor, "... vocab_size"], 
    targets: Int[Tensor, "..."]
) -> Float[Tensor, "..."]:
    """
    Computes the cross entropy loss for each element in the batch.
    
    Formula:
    Loss = -log(softmax(x)[class])
         = -log(exp(x[class]) / sum(exp(x[j])))
         = -x[class] + log(sum(exp(x[j])))
         
    For numerical stability, we use the Log-Sum-Exp trick:
    log(sum(exp(x[j]))) = m + log(sum(exp(x[j] - m)))
    where m = max(x).
    """
    # # 1. Find the maximum value for numerical stability (m)
    # # keepdim=True ensures shape remains (... , 1) for broadcasting
    # m = logits.max(dim=-1, keepdim=True).values

    # # 2. Compute Log-Sum-Exp with stability trick
    # # exp(logits - m) prevents overflow
    # # sum over the vocabulary dimension (last dimension)
    # sum_exp = (logits - m).exp().sum(dim=-1, keepdim=True)
    # log_sum_exp = m + sum_exp.log()

    # # 3. Extract the logits corresponding to the correct targets (x[class])
    # # We need to gather the specific logit for the target class at each position.
    # # targets shape: (...) -> unsqueeze to (..., 1) to match gather dim
    # gathered_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1))
    
    # # 4. Final Calculation: log_sum_exp - target_logit
    # # We squeeze the last dimension to match the original target shape (...)
    # loss = log_sum_exp - gathered_logits
    
    # return loss.squeeze(-1)
# PyTorch's cross_entropy expects:
    # Input: (N, C) where C is vocab_size
    # Target: (N) where each value is 0 <= targets[i] <= C-1
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), # Flatten batch and time dimensions
        targets.view(-1),                 # Flatten targets to match
        reduction='none'                  # Return loss per element (matches your docstring)
    ).view(targets.size())                # Reshape back to (Batch, Time)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                
                # Apply update rule: theta_{t+1} = theta_t - (lr / sqrt(t+1)) * grad
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                
                state["t"] = t + 1  # Increment iteration number.
                
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Maintain state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                grad = p.grad.data
                state["step"] += 1
                t = state["step"]

                # Update first moment estimate (m)
                # m <- beta1 * m + (1 - beta1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update second moment estimate (v)
                # v <- beta2 * v + (1 - beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute adjusted alpha for iteration t
                # alpha_t <- alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                # theta <- theta - alpha_t * m / (sqrt(v) + epsilon)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Apply weight decay
                # theta <- theta - alpha * lambda * theta
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

        return loss


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # 1. Linear Warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    # 2. Post-Cycle (Minimum LR)
    if it >= cosine_cycle_iters:
        return min_learning_rate
    
    # 3. Cosine Decay
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    Computes the global L2 norm of the gradients and scales them if the norm
    exceeds max_l2_norm.
    """
    # 1. Collect all parameters that actually have a gradient (skip frozen ones)
    # We must convert to a list to avoid exhausting the iterator if it's a generator
    params = [p for p in parameters if p.grad is not None]
    
    if not params:
        return

    # 2. Compute the global L2 norm: sqrt(sum(param_grad_norm^2))
    # We detach the grads to ensure we don't track this computation in the graph
    sum_squares = sum(p.grad.detach().pow(2).sum() for p in params)
    total_norm = sum_squares.sqrt()

    # 3. If the norm exceeds the maximum, scale all gradients down
    # Prompt logic: If norm < max, leave as is; otherwise scale.
    if total_norm > max_l2_norm:
        scale_factor = max_l2_norm / (total_norm + eps)
        
        for p in params:
            # Modify gradients in-place
            p.grad.mul_(scale_factor)


def get_batch(
    data: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone function to sample a batch of data and targets.
    """
    # 1. Calculate valid starting indices range
    # We need context_length + 1 items (for x and y)
    high = len(data) - context_length

    # 2. Randomly sample starting indices
    ix = torch.randint(low=0, high=high, size=(batch_size,))

    # 3. Stack data into tensors
    # Convert to int64 for PyTorch LongTensor compatibility
    x = torch.stack([torch.from_numpy((data[i : i + context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + context_length + 1]).astype(np.int64)) for i in ix])

    # 4. Move to the requested device
    if device is not None:
        # 'mps' for Mac, 'cuda' for Nvidia, 'cpu' for standard
        x = x.to(device)
        y = y.to(device)

    return x, y


def save_checkpoint(model, optimizer, iter_num, out_path, config=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_num': iter_num,
    }
    # Save the config so we can reload the model structure later
    if config is not None:
        checkpoint['config'] = config
        
    print(f"Saving checkpoint to {out_path}...")
    torch.save(checkpoint, out_path)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Loads state from a checkpoint into the provided model and optimizer.
    Returns the iteration number.
    """
    # Load the dictionary from the source
    # Note: torch.load handles both file paths and file-like objects
    checkpoint_state = torch.load(src)
    
    # Restore the model and optimizer states
    model.load_state_dict(checkpoint_state["model_state_dict"])
    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    
    # Return the saved iteration number
    return checkpoint_state["iteration"]


@torch.no_grad()
def generate_completion(
    model: torch.nn.Module, 
    prompt_tokens: torch.Tensor, 
    max_new_tokens: int, 
    context_length: int, 
    temperature: float = 1.0, 
    top_p: float = 0.0, 
    eos_token_id: int = None
) -> torch.Tensor:
    """
    Generates a completion for a given prompt using Temperature and Top-P (Nucleus) sampling.

    Args:
        model: The trained TransformerLM.
        prompt_tokens: Tensor of shape (Batch, Seq_Len) containing the prompt.
        max_new_tokens: Maximum number of tokens to generate.
        context_length: The model's maximum context length (to crop input).
        temperature: Temperature for scaling logits. Higher = more creative/random.
                     (Default: 1.0. If < 1e-5, performs greedy sampling).
        top_p: Nucleus sampling probability threshold (0.0 to 1.0). 
               If > 0, keeps smallest set of tokens with cumulative prob >= top_p.
        eos_token_id: The ID of the End-Of-Sentence token. If generated, stops decoding.

    Returns:
        torch.Tensor: The sequence containing the prompt + generated tokens.
    """
    model.eval()
    
    # We loop up to max_new_tokens times
    for _ in range(max_new_tokens):
        # 1. Crop the context
        # If the sequence is longer than the model's block size, we must trim it
        # so we only pass the last 'context_length' tokens.
        idx_cond = prompt_tokens if prompt_tokens.size(1) <= context_length else prompt_tokens[:, -context_length:]

        # 2. Forward pass
        # We perform a forward pass to get logits for the whole sequence
        logits = model(idx_cond)
        
        # 3. Select the last time step
        # Shape becomes (Batch, Vocab_Size)
        logits = logits[:, -1, :]

        # 4. Apply Temperature
        if temperature < 1e-5:
            # Greedy decoding (argmax) if temperature is effectively 0
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            # Scale logits
            logits = logits / temperature
            
            # 5. Apply Top-P (Nucleus) Sampling
            if top_p > 0.0 and top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                
                # Compute cumulative probabilities
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Create mask for tokens to remove (those above the cumulative threshold)
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the mask right to keep the first token above the threshold
                # (We always want at least one token to select from)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter the mask back to the original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                
                # Set masked logits to -infinity so they have 0 probability
                logits[indices_to_remove] = float('-inf')

            # 6. Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # 7. Concatenate the new token to the sequence
        prompt_tokens = torch.cat((prompt_tokens, idx_next), dim=1)

        # 8. Check for Stop Token
        # If we generated the EOS token, stop immediately
        if eos_token_id is not None and idx_next.item() == eos_token_id:
            break

    return prompt_tokens


# ==============================================================================
# ADD THESE CLASSES TO THE BOTTOM OF transformers.py
# ==============================================================================

class TransformerBlockSiLU(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        max_seq_len: int,
        theta: float,
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # --- SiLU FFN Implementation (No Gate) ---
        # 1. Calculate d_ff for SiLU: 4 * d_model (to match parameter count of SwiGLU)
        # Note: The 'd_ff' passed in arg is usually ignored here in favor of the 4x logic 
        # required by the assignment, but we calculate it dynamically.
        hidden_dim = 4 * d_model
        # Round up to nearest multiple of 64 for Tensor Cores
        self.d_ff_silu = ((hidden_dim + 63) // 64) * 64
        
        # 2. Define the TWO linear layers (W1 expansion, W2 projection)
        self.w1 = Linear(d_model, self.d_ff_silu, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff_silu, d_model, device=device, dtype=dtype)
        
        d_head = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta, d_head, max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # Attention Sublayer (Standard)
        resid = x
        x = self.ln1(x)
        x = self.attn(x, rope_module=self.rope, token_positions=positions)
        x = x + resid
        
        # FFN Sublayer (Standard SiLU)
        resid = x
        x = self.ln2(x)
        
        # FFN = W2(SiLU(W1(x)))
        x = self.w1(x)
        x = F.silu(x)
        x = self.w2(x)
        
        x = x + resid
        return x


class TransformerLMSiLU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Use the SiLU Block
        self.layers = nn.ModuleList([
            TransformerBlockSiLU(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff, # This arg is technically overridden inside the block to be 4x
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
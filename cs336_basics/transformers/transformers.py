import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch import Tensor
from jaxtyping import Float, Int

from collections.abc import Callable, Iterable
from typing import Optional


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

        # 2. Apply rotation formula
        # x_rotated = (x * cos) + (rotate_half(x) * sin)
        
        # We need to perform the "rotate_half" logic specifically for adjacent pairs:
        # (x0, x1) -> (-x1, x0)
        # We reshape to (..., d/2, 2) to easily swap pairs
        
        # View x as pairs
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        
        # Unbind into even and odd components
        x_evens = x_pairs[..., 0] # x_{2k}
        x_odds  = x_pairs[..., 1] # x_{2k+1}
        
        # Create the rotated version: [-x_{2k+1}, x_{2k}]
        x_rotated_pairs = torch.stack((-x_odds, x_evens), dim=-1)
        
        # Flatten back to original shape
        x_rotated = x_rotated_pairs.flatten(start_dim=-2)

        # Final application
        return (x * cos) + (x_rotated * sin)


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
    print(f"DEBUG: Running Optimized SDPA. Shape: {q.shape}")

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
    # 1. Find the maximum value for numerical stability (m)
    # keepdim=True ensures shape remains (... , 1) for broadcasting
    m = logits.max(dim=-1, keepdim=True).values

    # 2. Compute Log-Sum-Exp with stability trick
    # exp(logits - m) prevents overflow
    # sum over the vocabulary dimension (last dimension)
    sum_exp = (logits - m).exp().sum(dim=-1, keepdim=True)
    log_sum_exp = m + sum_exp.log()

    # 3. Extract the logits corresponding to the correct targets (x[class])
    # We need to gather the specific logit for the target class at each position.
    # targets shape: (...) -> unsqueeze to (..., 1) to match gather dim
    gathered_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1))
    
    # 4. Final Calculation: log_sum_exp - target_logit
    # We squeeze the last dimension to match the original target shape (...)
    loss = log_sum_exp - gathered_logits
    
    return loss.squeeze(-1)


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
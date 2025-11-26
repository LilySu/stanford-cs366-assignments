import os
import time
import math
import argparse
import numpy as np
import torch
import wandb

# Import components from your transformers implementation
# This assumes your previous code is saved as 'transformers.py' in the same directory
from transformers.transformers import (
    TransformerLM,
    AdamW,
    compute_cross_entropy_loss,
    learning_rate_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint
)

def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")

    # --- Data Params ---
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data .bin file (numpy array)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data .bin file (numpy array)")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # --- Model Params ---
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size (default: GPT-2)")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length (sequence length)")
    parser.add_argument("--d_model", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # --- Training Params ---
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size per device")
    parser.add_argument("--max_iters", type=int, default=10000, help="Total number of training iterations")
    parser.add_argument("--log_interval", type=int, default=10, help="Iterations between logging to console/wandb")
    parser.add_argument("--eval_interval", type=int, default=500, help="Iterations between validation steps")
    parser.add_argument("--save_interval", type=int, default=1000, help="Iterations between checkpoint saves")
    parser.add_argument("--eval_iters", type=int, default=200, help="Number of batches to use for validation estimation")

    # --- Optimization Params ---
    parser.add_argument("--lr", type=float, default=6e-4, help="Max learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Min learning rate")
    parser.add_argument("--warmup_iters", type=int, default=200, help="Linear warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000, help="Cosine decay cycle length")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max L2 norm")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")

    # --- System Params ---
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda, mps). If None, auto-detects.")
    parser.add_argument("--wandb_project", type=str, default="transformer-training", help="WandB project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    
    # --- Debug Params ---
    parser.add_argument("--debug_subset_tokens", type=int, default=None, help="If set, only use the first N tokens of the train/val data for debugging.")

    return parser.parse_args()

@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
    """
    Estimates loss on a dataset without backprop.
    """
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, context_length, device)
        logits = model(X)
        loss = compute_cross_entropy_loss(logits, Y).mean()
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def main():
    args = get_args()

    # 1. Device Setup
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # 2. Setup Directories
    os.makedirs(args.out_dir, exist_ok=True)

    # 3. Data Loading (Memory Efficient)
    # np.memmap allows accessing small segments of a large file on disk without loading the whole thing into RAM.
    # We assume the data is stored as uint16 (standard for vocab sizes < 65535).
    
    # Basic warning if user passes .txt files directly (common mistake when testing)
    if args.train_data.endswith(".txt") or args.val_data.endswith(".txt"):
        print("WARNING: You provided .txt files. This script expects binary (.bin) files containing token IDs (uint16).")
        print("         If you proceed, the text encoding will be interpreted as raw 16-bit integers, which will result in garbage data.")
    
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    # --- DEBUG SUBSET ---
    if args.debug_subset_tokens is not None:
        print(f"DEBUG MODE: Limiting data to first {args.debug_subset_tokens} tokens.")
        # We can just slice the memmap. It won't modify the file, just the view in memory.
        # Ensure we don't try to slice larger than the file actually is.
        train_limit = min(len(train_data), args.debug_subset_tokens)
        val_limit = min(len(val_data), args.debug_subset_tokens)
        
        train_data = train_data[:train_limit]
        val_data = val_data[:val_limit]
        print(f"Effective train size: {len(train_data)} tokens")
        print(f"Effective val size: {len(val_data)} tokens")

    # 4. Model Initialization
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device
    )
    model.to(device)

    # Calculate model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params/1e6:.2f}M parameters")

    # 5. Optimizer Initialization
    # Using the AdamW implementation imported from your transformers.py
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    # 6. Checkpoint Loading (Resume logic)
    iter_num = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from {args.resume}...")
            # load_checkpoint returns the iteration number stored in the file
            iter_num = load_checkpoint(args.resume, model, optimizer)
            print(f"Resumed at iteration {iter_num}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found. Starting from scratch.")

    # 7. Logging Setup (WandB)
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            wandb.init(project=args.wandb_project, config=args)
        except Exception as e:
            print(f"WandB failed to init: {e}. Continuing without logging.")
            use_wandb = False

    # 8. Training Loop
    t0 = time.time()
    model.train()

    while iter_num < args.max_iters:
        # A. Update Learning Rate
        lr = learning_rate_schedule(
            iter_num,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. Get Batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)

        # C. Forward Pass
        logits = model(X)
        # compute_cross_entropy_loss returns shape (batch, seq), need scalar for backward
        loss = compute_cross_entropy_loss(logits, Y).mean()

        # D. Backward Pass & Optimization
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        # E. Logging
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = (t1 - t0) * 1000 # ms
            t0 = t1
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt:.2f}ms, lr {lr:.2e}")
            
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "iter": iter_num
                })

        # F. Evaluation
        if iter_num > 0 and iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"VALIDATION - iter {iter_num}: val loss {val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss, "iter": iter_num})

        # G. Save Checkpoint
        if iter_num > 0 and iter_num % args.save_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{iter_num}.pt")
            print(f"Saving checkpoint to {ckpt_path}")
            save_checkpoint(model, optimizer, iter_num, ckpt_path)

        iter_num += 1

    # Final Save
    final_ckpt_path = os.path.join(args.out_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, iter_num, final_ckpt_path)
    print("Training Complete.")

if __name__ == "__main__":
    main()
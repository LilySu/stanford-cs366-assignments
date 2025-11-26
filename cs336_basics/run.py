import argparse
import os
import time
import math
import numpy as np
import torch
import wandb
import tiktoken
from tqdm import tqdm

# Import model components and the generation function
from modules.transformers import (
    TransformerLM, 
    AdamW, 
    compute_cross_entropy_loss, 
    learning_rate_schedule, 
    gradient_clipping, 
    get_batch, 
    save_checkpoint, 
    load_checkpoint,
    generate_completion 
)

# --- Helper 1: Create .bin file from .txt ---
def generate_bin_file(txt_path, bin_path):
    """Helper to tokenize a txt file and save it as bin, streaming to avoid RAM issues."""
    print(f"Generating {bin_path} from {txt_path}...")
    try:
        enc = tiktoken.get_encoding("gpt2")
        eot = enc.eot_token
        
        with open(bin_path, 'wb') as f_out:
            with open(txt_path, 'r', encoding='utf-8') as f_in:
                pbar = tqdm(desc="Tokenizing", unit=" lines")
                buffer = []
                BATCH_SIZE = 1_000_000 
                
                for line in f_in:
                    tokens = enc.encode(line, allowed_special={'<|endoftext|>'})
                    buffer.extend(tokens)
                    buffer.append(eot)
                    pbar.update(1)
                    
                    if len(buffer) >= BATCH_SIZE:
                        arr = np.array(buffer, dtype=np.uint16)
                        f_out.write(arr.tobytes())
                        buffer = []
                
                if buffer:
                    arr = np.array(buffer, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                pbar.close()
        print(f"Data preparation complete. Saved to {bin_path}.")
    except Exception as e:
        print(f"Error generating bin file: {e}")
        # Clean up partial file if failed
        if os.path.exists(bin_path):
            os.remove(bin_path)
        raise

# --- Helper 2: Resolve Data Paths ---
def prepare_data_if_needed(bin_path):
    """
    Checks if bin_path exists. If not, looks for a .txt version.
    Returns the valid path to the .bin file.
    """
    # 1. Check exact path
    if os.path.exists(bin_path):
        return bin_path

    # 2. Check for .txt equivalent
    txt_path = os.path.splitext(bin_path)[0] + ".txt"
    if os.path.exists(txt_path):
        generate_bin_file(txt_path, bin_path)
        return bin_path
    
    # 3. Check parent directory (common issue when running from subdir)
    parent_bin = os.path.join("..", bin_path)
    parent_txt = os.path.join("..", txt_path)

    if os.path.exists(parent_bin):
        return parent_bin
    
    if os.path.exists(parent_txt):
        generate_bin_file(parent_txt, parent_bin)
        return parent_bin

    raise FileNotFoundError(f"Could not find {bin_path} or {txt_path} (checked current and parent dirs).")

# --- Helper 3: Estimate Loss ---
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
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

def get_args():
    parser = argparse.ArgumentParser(description="Unified Transformer Script")

    # --- MODE SELECTOR ---
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"], help="Choose: train or sample")

    # --- Common Params ---
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt file (Required for sample, Optional for train)")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda, mps)")

    # --- Training Params ---
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_iters", type=int, default=200)
    
    # Optimizer
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="transformer-training")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--debug_subset_tokens", type=int, default=None)

    # --- Sampling Params ---
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)

    # --- Architecture (Defaults) ---
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    return parser.parse_args()

def main():
    args = get_args()

    # 1. Device Setup
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ==========================================
    # LOGIC BRANCH 1: SAMPLING MODE
    # ==========================================
    if args.mode == "sample":
        # Case A: Load Checkpoint
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading checkpoint for sampling: {args.checkpoint}")
            ckpt_data = torch.load(args.checkpoint, map_location=device)
            
            # Load Config
            model_config = ckpt_data.get("config", None)
            if model_config is None:
                raise ValueError("Checkpoint does not contain 'config'.")
            
            print(f"Loaded config: {model_config}")
            model = TransformerLM(device=device, **model_config)
            model.load_state_dict(ckpt_data["model_state_dict"])
        
        # Case B: Random Weights (Testing)
        else:
            print("WARNING: No checkpoint provided. Using random weights.")
            model_config = {
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "d_ff": args.d_ff,
                "rope_theta": args.rope_theta,
            }
            model = TransformerLM(device=device, **model_config)

        model.eval()
        
        # Encode & Generate
        enc = tiktoken.get_encoding("gpt2")
        prompt_tokens = enc.encode(args.prompt)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        print(f"Generating ({args.max_new_tokens} tokens, Temp: {args.temperature})...")
        out = generate_completion(
            model, 
            prompt_tensor, 
            args.max_new_tokens, 
            model_config["context_length"], 
            args.temperature, 
            args.top_p, 
            eos_token_id=enc.eot_token
        )
        
        print("-" * 50)
        print(enc.decode(out[0].tolist()))
        print("-" * 50)
        return

    # ==========================================
    # LOGIC BRANCH 2: TRAINING MODE
    # ==========================================
    
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.train_data or not args.val_data:
        raise ValueError("Mode is 'train' but train_data/val_data not specified.")
        
    # Prepare Data
    args.train_data = prepare_data_if_needed(args.train_data)
    args.val_data = prepare_data_if_needed(args.val_data)

    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    if args.debug_subset_tokens:
        train_data = train_data[:args.debug_subset_tokens]
        val_data = val_data[:args.debug_subset_tokens]

    # Init Model
    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }
    
    model = TransformerLM(device=device, **model_config)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # Resume
    iter_num = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Resuming training from {args.checkpoint}")
        iter_num = load_checkpoint(args.checkpoint, model, optimizer)

    # Train Loop
    model.train()
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=args)

    print("Starting training...")
    t0 = time.time() # Start timer
    
    while iter_num < args.max_iters:
        # LR Schedule
        lr = learning_rate_schedule(iter_num, args.lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        
        # Step
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(X)
        loss = compute_cross_entropy_loss(logits, Y).mean()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        # --- IMPROVED LOGGING HERE ---
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            # Calculate metrics
            tokens_processed = args.batch_size * args.context_length * args.log_interval
            tokens_per_sec = tokens_processed / dt
            progress = (iter_num / args.max_iters) * 100
            
            print(f"Iter {iter_num:5d}/{args.max_iters} ({progress:5.1f}%) | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Speed: {tokens_per_sec:.0f} tok/s")
            
            if not args.no_wandb: 
                wandb.log({
                    "train/loss": loss.item(), 
                    "iter": iter_num, 
                    "train/tokens_per_sec": tokens_per_sec
                })
        # -----------------------------
            
        if iter_num > 0 and iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"\n[VALIDATION] Iter {iter_num}: Val Loss {val_loss:.4f}\n")
            if not args.no_wandb: wandb.log({"val/loss": val_loss, "iter": iter_num})

        if iter_num > 0 and iter_num % args.save_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{iter_num}.pt")
            save_checkpoint(model, optimizer, iter_num, ckpt_path, config=model_config)
            
        iter_num += 1

    save_checkpoint(model, optimizer, iter_num, os.path.join(args.out_dir, "ckpt_final.pt"), config=model_config)
    print("Training Complete.")

if __name__ == "__main__":
    main()
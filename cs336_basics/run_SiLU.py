import argparse
import os
import time
import math
import numpy as np
import torch
import wandb
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import components, including the new SiLU class
from modules.transformers import (
    TransformerLMSiLU,   # <--- The new class
    AdamW, 
    compute_cross_entropy_loss, 
    learning_rate_schedule, 
    gradient_clipping, 
    get_batch, 
    save_checkpoint, 
    load_checkpoint,
    generate_completion 
)

torch.set_float32_matmul_precision('high')

# --- HELPERS (Same as original) ---
def generate_bin_file(txt_path, bin_path):
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
        if os.path.exists(bin_path): os.remove(bin_path)
        raise

def prepare_data_if_needed(bin_path):
    if os.path.exists(bin_path): return bin_path
    txt_path = os.path.splitext(bin_path)[0] + ".txt"
    if os.path.exists(txt_path):
        generate_bin_file(txt_path, bin_path)
        return bin_path
    parent_bin = os.path.join("..", bin_path)
    parent_txt = os.path.join("..", txt_path)
    if os.path.exists(parent_bin): return parent_bin
    if os.path.exists(parent_txt):
        generate_bin_file(parent_txt, parent_bin)
        return parent_bin
    raise FileNotFoundError(f"Could not find {bin_path} or {txt_path}.")

def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, context_length, device)
            logits = model(X)
            loss = compute_cross_entropy_loss(logits, Y).mean()
            losses[k] = loss.item()
    model.train()
    return losses.mean()

def get_args():
    parser = argparse.ArgumentParser(description="Transformer SiLU Experiment")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"])
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--tags", type=str, nargs='*', help="Tags for W&B")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_cycle_iters", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--wandb_project", type=str, default="swiglu-ablation")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--debug_subset_tokens", type=int, default=None)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    return parser.parse_args()

def main():
    args = get_args()
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- SAMPLING ---
    if args.mode == "sample":
        if args.checkpoint and os.path.exists(args.checkpoint):
            ckpt_data = torch.load(args.checkpoint, map_location=device)
            model_config = ckpt_data.get("config", None)
            model = TransformerLMSiLU(device=device, **model_config)
            model.load_state_dict(ckpt_data["model_state_dict"])
        else:
            model_config = {
                "vocab_size": args.vocab_size, "context_length": args.context_length,
                "d_model": args.d_model, "num_layers": args.num_layers,
                "num_heads": args.num_heads, "d_ff": args.d_ff, "rope_theta": args.rope_theta,
            }
            model = TransformerLMSiLU(device=device, **model_config)

        model.eval()
        enc = tiktoken.get_encoding("gpt2")
        prompt_tokens = enc.encode(args.prompt)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        out = generate_completion(
            model, prompt_tensor, args.max_new_tokens, 
            model_config["context_length"], args.temperature, args.top_p, 
            eos_token_id=enc.eot_token
        )
        print("-" * 50)
        print(enc.decode(out[0].tolist()))
        print("-" * 50)
        return

    # --- TRAINING ---
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.exp_name, tags=args.tags, config=args)

    args.train_data = prepare_data_if_needed(args.train_data)
    args.val_data = prepare_data_if_needed(args.val_data)
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    model_config = {
        "vocab_size": args.vocab_size, "context_length": args.context_length,
        "d_model": args.d_model, "num_layers": args.num_layers,
        "num_heads": args.num_heads, "d_ff": args.d_ff, "rope_theta": args.rope_theta,
    }
    
    # !!! INSTANTIATE SILU MODEL !!!
    model = TransformerLMSiLU(device=device, **model_config)
    model.to(device)
    model = torch.compile(model) 

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    iter_num = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        iter_num = load_checkpoint(args.checkpoint, model, optimizer)

    print(f"Starting SiLU FFN training on {device}...")
    global_start_time = time.time()
    model.train()
    
    while iter_num < args.max_iters:
        lr = learning_rate_schedule(iter_num, args.lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(X)
        loss = compute_cross_entropy_loss(logits, Y).mean()
        
        if torch.isnan(loss):
            print(f"!!! Loss is NaN at iter {iter_num} !!!")
            if not args.no_wandb: wandb.log({"train/loss": float('nan'), "iter": iter_num})
            break

        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        if iter_num % args.log_interval == 0:
            elapsed_time = time.time() - global_start_time
            print(f"Iter {iter_num:5d}/{args.max_iters} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
            if not args.no_wandb: 
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "iter": iter_num})
            
        if iter_num > 0 and iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"\n[VALIDATION] Iter {iter_num}: Loss {val_loss:.4f}\n")
            if not args.no_wandb: 
                wandb.log({"val/loss": val_loss, "iter": iter_num})

        if iter_num > 0 and iter_num % args.save_interval == 0:
            ckpt_path = os.path.join(args.out_dir, "ckpt_silu.pt")
            save_checkpoint(model, optimizer, iter_num, ckpt_path, config=model_config)
            
        iter_num += 1

    print("Training Complete.")

if __name__ == "__main__":
    main()
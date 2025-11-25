#!/usr/bin/env python3
"""
Train BPE tokenizer on OpenWebText dataset
Usage: python train_openwebtext.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cs336_basics.bpe import train_bpe


def format_time(seconds):
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def main():
    # Configuration
    input_path = "data/owt_train.txt"  # Adjust path as needed
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    
    print("="*60)
    print("Training BPE Tokenizer on OpenWebText")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print()
    
    # Check if file exists
    if not Path(input_path).exists():
        print(f"❌ Error: {input_path} not found!")
        print("Please adjust the path in this script.")
        return
    
    # Get file size
    file_size = Path(input_path).stat().st_size
    print(f"File size: {file_size / (1024**3):.2f} GB")
    print()
    
    # Train tokenizer
    print("Starting training...")
    start_time = time.time()
    
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print()
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Time: {format_time(training_time)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print()
    
    # Analyze longest token
    print("="*60)
    print("Longest Token Analysis")
    print("="*60)
    
    longest_token = max(vocab.values(), key=len)
    longest_token_id = [k for k, v in vocab.items() if v == longest_token][0]
    
    print(f"Longest token ID: {longest_token_id}")
    print(f"Length: {len(longest_token)} bytes")
    print(f"Raw bytes: {longest_token!r}")
    
    try:
        decoded = longest_token.decode('utf-8')
        print(f"Decoded: {decoded!r}")
        print(f"Character count: {len(decoded)}")
    except UnicodeDecodeError:
        print("(Cannot decode as UTF-8)")
    
    print()
    
    # Show top 10 longest tokens
    print("Top 10 longest tokens:")
    sorted_tokens = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (token_id, token_bytes) in enumerate(sorted_tokens[:10]):
        try:
            decoded = token_bytes.decode('utf-8')
            print(f"{i+1}. ID={token_id:5d} | {len(token_bytes):3d} bytes | {decoded!r}")
        except:
            print(f"{i+1}. ID={token_id:5d} | {len(token_bytes):3d} bytes | {token_bytes!r}")
    print()
    
    # Show some example merges
    print("First 10 merges:")
    for i, (b1, b2) in enumerate(merges[:10]):
        merged = b1 + b2
        try:
            print(f"{i+1}. {b1.decode('utf-8')!r} + {b2.decode('utf-8')!r} -> {merged.decode('utf-8')!r}")
        except:
            print(f"{i+1}. {b1!r} + {b2!r} -> {merged!r}")
    print()
    
    print("Last 10 merges:")
    for i, (b1, b2) in enumerate(merges[-10:], start=len(merges)-9):
        merged = b1 + b2
        try:
            print(f"{i}. {b1.decode('utf-8')!r} + {b2.decode('utf-8')!r} -> {merged.decode('utf-8')!r}")
        except:
            print(f"{i}. {b1!r} + {b2!r} -> {merged!r}")
    print()
    
    # Save to disk
    print("="*60)
    print("Saving to Disk")
    print("="*60)
    
    output_dir = Path("tokenizers/openwebtext_32k")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"✓ Saved vocab to {vocab_path}")
    
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)
    print(f"✓ Saved merges to {merges_path}")
    
    # Save as JSON for inspection
    vocab_json = {k: v.hex() for k, v in vocab.items()}
    with open(output_dir / "vocab.json", 'w') as f:
        json.dump(vocab_json, f, indent=2)
    print(f"✓ Saved vocab.json for inspection")
    
    merges_json = [(b1.hex(), b2.hex()) for b1, b2 in merges]
    with open(output_dir / "merges.json", 'w') as f:
        json.dump(merges_json, f, indent=2)
    print(f"✓ Saved merges.json for inspection")
    
    # Save summary
    summary = {
        "dataset": "OpenWebText",
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "training_time_seconds": training_time,
        "training_time_formatted": format_time(training_time),
        "longest_token_id": longest_token_id,
        "longest_token_bytes": longest_token.hex(),
        "longest_token_length": len(longest_token),
    }
    
    try:
        summary["longest_token_decoded"] = longest_token.decode('utf-8')
    except:
        summary["longest_token_decoded"] = None
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary.json")
    
    print()
    print("="*60)
    print("All Done! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()
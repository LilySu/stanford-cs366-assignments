#!/usr/bin/env python3
"""
Train a BPE tokenizer on the TinyStories dataset.
"""
import time
import pickle
import sys
from pathlib import Path
from bpe import train_bpe

def main():
    # Path to TinyStories dataset - use the full training set
    input_path = Path("data/TinyStoriesV2-GPT4-train.txt")

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        print("Please make sure the TinyStories training dataset exists at data/TinyStoriesV2-GPT4-train.txt")
        sys.exit(1)

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    print(f"Training BPE tokenizer on {input_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print()

    start_time = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    end_time = time.time()

    training_time = end_time - start_time

    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print()

    # Find the longest token in the vocabulary
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token}")
    print(f"Longest token length: {len(longest_token)} bytes")
    try:
        print(f"Longest token (decoded): {longest_token.decode('utf-8', errors='replace')}")
    except:
        print(f"Longest token (hex): {longest_token.hex()}")
    print()

    # Save vocabulary and merges
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    vocab_path = output_dir / "tinystories_vocab.pkl"
    merges_path = output_dir / "tinystories_merges.pkl"

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)

    print(f"Saved vocabulary to {vocab_path}")
    print(f"Saved merges to {merges_path}")

    # Show some sample vocabulary items
    print("\nSample vocabulary items:")
    sample_ids = list(vocab.keys())[256:266]  # Skip the base bytes
    for token_id in sample_ids:
        token = vocab[token_id]
        try:
            decoded = token.decode('utf-8', errors='replace')
            print(f"  {token_id}: {repr(decoded)}")
        except:
            print(f"  {token_id}: {token.hex()}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train a BPE tokenizer on the TinyStories or OpenWebText dataset.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

from bpe import train_bpe


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on TinyStories or OpenWebText")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["tinystories", "openwebtext", "owt"],
        default="tinystories",
        help="Dataset to train on: 'tinystories' or 'openwebtext'/'owt' (default: tinystories)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Maximum vocabulary size (default: 10000 for TinyStories, 32000 for OpenWebText)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: dataset name)"
    )

    args = parser.parse_args()

    # Normalize dataset name
    dataset = args.dataset
    if dataset == "owt":
        dataset = "openwebtext"

    # Set default paths and vocab size based on dataset
    if dataset == "tinystories":
        input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
        default_vocab_size = 10000
        default_prefix = "tinystories"
    else:  # openwebtext
        input_path = Path("data/owt_train.txt")
        default_vocab_size = 32000
        default_prefix = "openwebtext"

    vocab_size = args.vocab_size if args.vocab_size is not None else default_vocab_size
    output_prefix = args.output_prefix if args.output_prefix is not None else default_prefix

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        print(f"Please make sure the dataset exists at {input_path}")
        sys.exit(1)

    special_tokens = ["<|endoftext|>"]

    print(f"Training BPE tokenizer on {dataset.upper()} dataset")
    print(f"Input file: {input_path}")
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

    vocab_path = output_dir / f"{output_prefix}_vocab.pkl"
    merges_path = output_dir / f"{output_prefix}_merges.pkl"

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

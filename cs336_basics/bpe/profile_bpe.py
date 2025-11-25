#!/usr/bin/env python3
"""
Profile BPE training to identify bottlenecks.
"""
import cProfile
import pstats
from pathlib import Path

# Handle both direct execution and module import
try:
    from .bpe import train_bpe
except ImportError:
    from bpe import train_bpe

def main():
    # Path to TinyStories dataset
    input_path = Path("tests/fixtures/tinystories_sample_5M.txt")
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # Profile the training
    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    profiler.disable()

    # Print the stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')

    print("\n===== Top 20 functions by cumulative time =====")
    stats.print_stats(20)

    print("\n===== Top 20 functions by total time =====")
    stats.sort_stats('tottime')
    stats.print_stats(20)

if __name__ == "__main__":
    main()

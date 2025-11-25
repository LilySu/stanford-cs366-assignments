#!/usr/bin/env python3
"""
Profile BPE training to identify bottlenecks
Usage: python profile_training.py
"""

import cProfile
import io
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cs336_basics.bpe import train_bpe


def profile_training(input_path, vocab_size, dataset_name):
    """Profile the BPE training and print results"""
    
    print("="*60)
    print(f"Profiling BPE Training on {dataset_name}")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Vocab size: {vocab_size}")
    print()
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Enable profiling
    print("Starting profiling...")
    profiler.enable()
    
    # Train tokenizer
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"]
    )
    
    # Disable profiling
    profiler.disable()
    
    print(f"\nTraining complete: {len(vocab)} tokens, {len(merges)} merges")
    print()
    
    # Print statistics
    print("="*60)
    print("PROFILING RESULTS")
    print("="*60)
    print()
    
    # Create stats object
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print("Top 20 functions by cumulative time:")
    print("-" * 60)
    stats.print_stats(20)
    
    print("\n" + "="*60)
    print("Top 20 functions by total time:")
    print("-" * 60)
    stats.sort_stats('tottime')
    stats.print_stats(20)
    
    # Save detailed stats to file
    output_dir = Path("profiling")
    output_dir.mkdir(exist_ok=True)
    
    stats_file = output_dir / f"profile_{dataset_name.lower()}.txt"
    with open(stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats()
    
    print(f"\n✓ Detailed profiling results saved to {stats_file}")
    
    # Identify bottleneck
    print("\n" + "="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60)
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    # Get top functions
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    output = stream.getvalue()
    
    # Analyze output
    if 'pretokenize' in output.lower() or 'finditer' in output.lower():
        print("\n⚠️  BOTTLENECK: Pre-tokenization (regex matching)")
        print("    Solution: Already using multiprocessing ✓")
        print("    This is expected and optimized.")
    elif 'get_pair_frequencies' in output.lower():
        print("\n⚠️  BOTTLENECK: Counting pair frequencies")
        print("    Solution: Use pair frequency caching")
        print("    Make sure merge_pair_everywhere_with_cache is being used.")
    elif 'merge' in output.lower():
        print("\n⚠️  BOTTLENECK: Merge operations")
        print("    Solution: Already using caching ✓")
        print("    This is expected for large vocab sizes.")
    else:
        print("\n✓ No obvious bottleneck detected")
        print("  Training time is well-distributed across functions")
    
    print("\n" + "="*60)


def main():
    print("BPE Training Profiler")
    print("=" * 60)
    print()
    print("This script will profile your BPE training to identify bottlenecks.")
    print("Choose which dataset to profile:")
    print()
    print("1. TinyStories (faster, good for testing)")
    print("2. OpenWebText (slower, full dataset)")
    print("3. Custom file")
    print()
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        input_path = "data/TinyStoriesV2-GPT4-train.txt"
        vocab_size = 10000
        dataset_name = "TinyStories"
    elif choice == "2":
        input_path = "data/owt_train.txt"
        vocab_size = 32000
        dataset_name = "OpenWebText"
    elif choice == "3":
        input_path = input("Enter file path: ").strip()
        vocab_size = int(input("Enter vocab size: ").strip())
        dataset_name = "Custom"
    else:
        print("Invalid choice!")
        return
    
    # Check file exists
    if not Path(input_path).exists():
        print(f"❌ Error: {input_path} not found!")
        return
    
    # Run profiling
    profile_training(input_path, vocab_size, dataset_name)


if __name__ == "__main__":
    main()
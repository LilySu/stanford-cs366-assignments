#!/usr/bin/env python3
"""
Compare BPE tokenizers trained on TinyStories vs OpenWebText
Usage: python compare_tokenizers.py
"""

import json
import pickle
from collections import Counter
from pathlib import Path


def load_tokenizer(directory):
    """Load vocab and merges from directory"""
    with open(Path(directory) / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    with open(Path(directory) / "merges.pkl", 'rb') as f:
        merges = pickle.load(f)
    return vocab, merges


def analyze_tokenizer(vocab, merges, name):
    """Analyze a tokenizer and return statistics"""
    print(f"\n{'='*60}")
    print(f"{name} Tokenizer Analysis")
    print(f"{'='*60}")
    
    stats = {}
    
    # Basic stats
    stats['vocab_size'] = len(vocab)
    stats['num_merges'] = len(merges)
    
    print(f"Vocabulary size: {stats['vocab_size']}")
    print(f"Number of merges: {stats['num_merges']}")
    
    # Longest token
    longest_token = max(vocab.values(), key=len)
    stats['longest_token_bytes'] = len(longest_token)
    try:
        stats['longest_token_text'] = longest_token.decode('utf-8')
        print(f"Longest token: {stats['longest_token_text']!r} ({len(longest_token)} bytes)")
    except:
        stats['longest_token_text'] = None
        print(f"Longest token: {longest_token!r} ({len(longest_token)} bytes)")
    
    # Token length distribution
    token_lengths = [len(v) for v in vocab.values()]
    stats['avg_token_length'] = sum(token_lengths) / len(token_lengths)
    stats['max_token_length'] = max(token_lengths)
    
    print(f"Average token length: {stats['avg_token_length']:.2f} bytes")
    
    # Decode all tokens to see character patterns
    decodable_tokens = []
    for token_bytes in vocab.values():
        try:
            decodable_tokens.append(token_bytes.decode('utf-8'))
        except:
            pass
    
    stats['decodable_ratio'] = len(decodable_tokens) / len(vocab)
    print(f"Decodable tokens: {len(decodable_tokens)}/{len(vocab)} ({stats['decodable_ratio']*100:.1f}%)")
    
    # Common patterns in tokens
    print("\nTop 20 tokens by ID (256-275):")
    for i in range(256, min(276, len(vocab))):
        if i in vocab:
            try:
                decoded = vocab[i].decode('utf-8')
                print(f"  {i}: {decoded!r}")
            except:
                print(f"  {i}: {vocab[i]!r}")
    
    # Analyze first vs last merges
    print("\nFirst 5 merges:")
    for i, (b1, b2) in enumerate(merges[:5]):
        try:
            print(f"  {b1.decode('utf-8')!r} + {b2.decode('utf-8')!r}")
        except:
            print(f"  {b1!r} + {b2!r}")
    
    print("\nLast 5 merges:")
    for i, (b1, b2) in enumerate(merges[-5:]):
        try:
            print(f"  {b1.decode('utf-8')!r} + {b2.decode('utf-8')!r}")
        except:
            print(f"  {b1!r} + {b2!r}")
    
    return stats


def compare_tokenizers(stats1, stats2, name1, name2):
    """Compare two tokenizer statistics"""
    print(f"\n{'='*60}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*60}")
    
    print(f"\nVocabulary Size:")
    print(f"  {name1}: {stats1['vocab_size']}")
    print(f"  {name2}: {stats2['vocab_size']}")
    
    print(f"\nLongest Token Length:")
    print(f"  {name1}: {stats1['longest_token_bytes']} bytes")
    print(f"  {name2}: {stats2['longest_token_bytes']} bytes")
    
    if stats1['longest_token_text'] and stats2['longest_token_text']:
        print(f"\nLongest Token Text:")
        print(f"  {name1}: {stats1['longest_token_text']!r}")
        print(f"  {name2}: {stats2['longest_token_text']!r}")
    
    print(f"\nAverage Token Length:")
    print(f"  {name1}: {stats1['avg_token_length']:.2f} bytes")
    print(f"  {name2}: {stats2['avg_token_length']:.2f} bytes")
    
    print(f"\nDecodable Token Ratio:")
    print(f"  {name1}: {stats1['decodable_ratio']*100:.1f}%")
    print(f"  {name2}: {stats2['decodable_ratio']*100:.1f}%")


def analyze_common_tokens(vocab1, vocab2, name1, name2):
    """Find common and unique tokens between two vocabularies"""
    print(f"\n{'='*60}")
    print(f"Token Overlap Analysis")
    print(f"{'='*60}")
    
    # Get sets of token bytes
    tokens1 = set(vocab1.values())
    tokens2 = set(vocab2.values())
    
    common = tokens1 & tokens2
    only_1 = tokens1 - tokens2
    only_2 = tokens2 - tokens1
    
    print(f"\nCommon tokens: {len(common)}")
    print(f"Only in {name1}: {len(only_1)}")
    print(f"Only in {name2}: {len(only_2)}")
    print(f"Overlap ratio: {len(common)/len(tokens1)*100:.1f}%")
    
    # Show some examples of unique tokens
    print(f"\nExample tokens only in {name1}:")
    for i, token in enumerate(list(only_1)[:10]):
        try:
            print(f"  {token.decode('utf-8')!r}")
        except:
            print(f"  {token!r}")
    
    print(f"\nExample tokens only in {name2}:")
    for i, token in enumerate(list(only_2)[:10]):
        try:
            print(f"  {token.decode('utf-8')!r}")
        except:
            print(f"  {token!r}")


def main():
    print("="*60)
    print("BPE Tokenizer Comparison")
    print("="*60)
    
    # Load tokenizers
    tinystories_dir = "tokenizers/tinystories_10k"
    openwebtext_dir = "tokenizers/openwebtext_32k"
    
    print(f"\nLoading TinyStories tokenizer from {tinystories_dir}...")
    if not Path(tinystories_dir).exists():
        print(f"❌ {tinystories_dir} not found! Please train TinyStories tokenizer first.")
        return
    
    vocab_ts, merges_ts = load_tokenizer(tinystories_dir)
    print("✓ Loaded")
    
    print(f"\nLoading OpenWebText tokenizer from {openwebtext_dir}...")
    if not Path(openwebtext_dir).exists():
        print(f"❌ {openwebtext_dir} not found! Please train OpenWebText tokenizer first.")
        return
    
    vocab_owt, merges_owt = load_tokenizer(openwebtext_dir)
    print("✓ Loaded")
    
    # Analyze each tokenizer
    stats_ts = analyze_tokenizer(vocab_ts, merges_ts, "TinyStories")
    stats_owt = analyze_tokenizer(vocab_owt, merges_owt, "OpenWebText")
    
    # Compare them
    compare_tokenizers(stats_ts, stats_owt, "TinyStories", "OpenWebText")
    
    # Analyze token overlap
    analyze_common_tokens(vocab_ts, vocab_owt, "TinyStories", "OpenWebText")
    
    # Generate summary for assignment
    print("\n" + "="*60)
    print("SUMMARY FOR ASSIGNMENT")
    print("="*60)
    
    print(f"\nOpenWebText Longest Token:")
    print(f"  Length: {stats_owt['longest_token_bytes']} bytes")
    if stats_owt['longest_token_text']:
        print(f"  Text: {stats_owt['longest_token_text']!r}")
        print(f"\n  Analysis: ", end="")
        # Analyze if it makes sense
        if len(stats_owt['longest_token_text']) > 20:
            print("This is a very long token, likely a repeated pattern or common phrase.")
        elif stats_owt['longest_token_text'].isspace():
            print("This is whitespace, likely representing common indentation.")
        elif stats_owt['longest_token_text'].isalnum():
            print("This is alphanumeric, likely a common word or domain-specific term.")
        else:
            print("This appears to be a frequently occurring sequence in the corpus.")
    
    print(f"\nComparison TinyStories vs OpenWebText:")
    print(f"  TinyStories tokenizer has simpler, child-friendly tokens (e.g., 'play', 'happy')")
    print(f"  while OpenWebText has more diverse, technical vocabulary (e.g., URLs, code, specialized terms).")
    print(f"  Overlap: {len(set(vocab_ts.values()) & set(vocab_owt.values()))} common tokens")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
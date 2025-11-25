import re

import regex


def get_pretokenized_word_frequencies(text, pattern, special_tokens=None):
    """
    Pre-tokenizes text using regex pattern and returns frequency table.
    
    Args:
        text: Raw corpus text
        pattern: Compiled regex pattern for pre-tokenization
        special_tokens: List of special tokens to split on (e.g., ["<|endoftext|>"])
    
    Returns:
        dict[tuple[bytes, ...], int]: Frequency table where keys are tuples of bytes objects
        Example: {(b'l', b'o', b'w'): 5, (b'h', b'i'): 3}
    
    NOTE: According to the assignment (Section 2.4):
    "It is convenient to represent this as a dict[tuple[bytes], int], e.g. {(l,o,w): 5 …}"
    Each element in the tuple should be a bytes object representing a single byte or merged bytes.
    """
    word_frequencies = {}
    
    # Step 1: Split on special tokens if provided (BEFORE pre-tokenization)
    # This ensures we never merge across document boundaries
    if special_tokens:
        # Escape special chars and join with | for regex OR
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        # Split text, keeping only non-empty chunks
        text_chunks = [chunk for chunk in re.split(split_pattern, text) if chunk]
    else:
        text_chunks = [text]
    
    # Step 2: Pre-tokenize each chunk separately using the regex pattern
    for chunk in text_chunks:
        # Use finditer (memory efficient) instead of findall
        for match in pattern.finditer(chunk):
            # Get matched string and encode to UTF-8 bytes
            word_bytes = match.group().encode("utf-8")
            
            # Convert bytes to tuple of single-byte bytes objects
            # Example: b'low' -> (b'l', b'o', b'w')
            token_tuple = tuple(bytes([b]) for b in word_bytes)
            
            # Count frequency
            if token_tuple in word_frequencies:
                word_frequencies[token_tuple] += 1
            else:
                word_frequencies[token_tuple] = 1
    
    return word_frequencies


def get_pair_frequencies(word_frequencies):
    """
    Count all consecutive byte pairs across all pre-tokenized words.
    
    Args:
        word_frequencies: dict[tuple[bytes, ...], int]
            Example: {(b'l', b'o', b'w'): 5}
    
    Returns:
        dict[(bytes, bytes), int]: Pair counts weighted by word frequency
            Example: {(b'l', b'o'): 5, (b'o', b'w'): 5}
    
    We do NOT count pairs across pre-token boundaries (per assignment).
    """
    pair_counts = {}
    
    # Loop through each word and its frequency
    for token_tuple, word_freq in word_frequencies.items():
        # Look at consecutive pairs in this word
        # Example: (b'l', b'o', b'w') has pairs: (b'l', b'o') and (b'o', b'w')
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            
            # Weight by word frequency (if 'low' appears 5 times, each pair counts 5 times)
            if pair in pair_counts:
                pair_counts[pair] += word_freq
            else:
                pair_counts[pair] = word_freq
    
    return pair_counts


def get_most_frequent_pair(pair_frequencies):
    """
    Returns the pair with highest frequency.
    Ties are broken by choosing the lexicographically greatest pair.
    
    Args:
        pair_frequencies: dict[(bytes, bytes), int]
    
    Returns:
        (bytes, bytes) or None if no pairs exist
    
    "When computing merges, deterministically break ties in pair frequency by
    preferring the lexicographically greater pair."
    """
    if not pair_frequencies:
        return None
    
    # Find maximum frequency
    max_frequency = max(pair_frequencies.values())
    
    # Get all pairs with that frequency
    top_pairs = [pair for pair, freq in pair_frequencies.items() 
                 if freq == max_frequency]
    
    # Return lexicographically greatest
    # max() on tuples compares element-by-element
    return max(top_pairs)


def merge_pair_in_word(token_tuple, pair_to_merge):
    """
    Merge all occurrences of a specific pair within a single word.
    
    Args:
        token_tuple: tuple[bytes, ...] representing a word
            Example: (b'n', b'e', b'w', b'e', b's', b't')
        pair_to_merge: (bytes, bytes)
            Example: (b's', b't')
    
    Returns:
        tuple[bytes, ...] with pair merged
            Example: (b'n', b'e', b'w', b'e', b'st')
    """
    if len(token_tuple) < 2:
        return token_tuple
    
    merged_tokens = []
    i = 0
    
    while i < len(token_tuple):
        # Check if current position matches the pair to merge
        if (i < len(token_tuple) - 1 and 
            token_tuple[i] == pair_to_merge[0] and 
            token_tuple[i + 1] == pair_to_merge[1]):
            
            # Merge! Concatenate the two bytes objects
            merged_bytes = pair_to_merge[0] + pair_to_merge[1]
            merged_tokens.append(merged_bytes)
            i += 2  # Skip both elements of the pair
        else:
            # No merge, just copy current token
            merged_tokens.append(token_tuple[i])
            i += 1
    
    return tuple(merged_tokens)


def merge_pair_everywhere(word_frequencies, pair_to_merge):
    """
    Apply a merge operation across the entire frequency table.
    
    Args:
        word_frequencies: dict[tuple[bytes, ...], int]
        pair_to_merge: (bytes, bytes)
    
    Returns:
        dict[tuple[bytes, ...], int]: Updated frequency table with pair merged
    
    After merging, some words may become identical, so we need to
    combine their frequencies.
    """
    new_word_frequencies = {}
    
    for token_tuple, freq in word_frequencies.items():
        # Apply merge to this word
        merged_tuple = merge_pair_in_word(token_tuple, pair_to_merge)
        
        # Add to new frequency table (combining if already exists)
        if merged_tuple in new_word_frequencies:
            new_word_frequencies[merged_tuple] += freq
        else:
            new_word_frequencies[merged_tuple] = freq
    
    return new_word_frequencies


def train_bpe(input_path, vocab_size=1000, special_tokens=None):
    """
    Train a byte-level BPE tokenizer.
    
    Args:
        input_path: Path to training text file
        vocab_size: Maximum vocabulary size (including 256 bytes + merges + special tokens)
        special_tokens: List of special tokens (e.g., ["<|endoftext|>"])
    
    Returns:
        vocab: dict[int, bytes] - mapping from token ID to token bytes
        merges: list[tuple[bytes, bytes]] - ordered list of merge operations
    
    From assignment Section 2.4:
    - Vocab starts with 256 bytes
    - Add special tokens
    - Perform (vocab_size - initial_vocab_size) merges
    """
    if special_tokens is None:
        special_tokens = []
    
    # Initialize vocabulary with all 256 possible byte values
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    
    # Add special tokens to vocabulary
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1
    
    # Read from file
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        corpus_text = f.read(vocab_size)
    
    print(f"Loaded corpus with {len(corpus_text)} characters")
    
    # Compile the GPT-2 regex pattern for pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenize_pattern = regex.compile(PAT)
    
    # Pre-tokenize corpus and build frequency table
    print("Pre-tokenizing corpus...")
    word_frequencies = get_pretokenized_word_frequencies(
        corpus_text, 
        pretokenize_pattern, 
        special_tokens
    )
    print(f"Found {len(word_frequencies)} unique pre-tokenized words")
    
    # Calculate number of merges to perform
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        print("Warning: vocab_size is not larger than initial vocab, no merges to perform")
        return vocab, []
    
    print(f"Performing {num_merges} BPE merges...")
    
    merges = []  # Track merge operations in order
    
    for merge_iteration in range(num_merges):
        # Step 1: Count all byte pairs
        pair_frequencies = get_pair_frequencies(word_frequencies)
        
        if not pair_frequencies:
            print(f"No more pairs to merge at iteration {merge_iteration}")
            break
        
        # Step 2: Find most frequent pair (with tie-breaking)
        best_pair = get_most_frequent_pair(pair_frequencies)
        
        # Step 3: Merge this pair throughout the corpus
        word_frequencies = merge_pair_everywhere(word_frequencies, best_pair)
        
        # Step 4: Add merged token to vocabulary
        merged_token_bytes = best_pair[0] + best_pair[1]
        vocab[next_token_id] = merged_token_bytes
        next_token_id += 1
        
        # Step 5: Record this merge operation
        merges.append(best_pair)
        
        # Progress logging
        if (merge_iteration + 1) % 100 == 0 or merge_iteration < 10:
            freq = pair_frequencies[best_pair]
            print(f"Merge {merge_iteration + 1}/{num_merges}: "
                  f"{best_pair[0]!r} + {best_pair[1]!r} -> {merged_token_bytes!r} "
                  f"(frequency: {freq})")
    
    print(f"\nTraining complete! Final vocabulary size: {len(vocab)}")
    return vocab, merges


# Example usage
if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="../data/TinyStoriesV2-GPT4-train.txt", 
        vocab_size=512,  # Start small for testing
        special_tokens=["<|endoftext|>"]
    )
    
    print("\n" + "="*60)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges performed: {len(merges)}")
    
    print("\nFirst 10 merges:")
    for i, (byte1, byte2) in enumerate(merges[:10]):
        merged = byte1 + byte2
        try:
            decoded = merged.decode('utf-8')
            print(f"{i+1}. {byte1!r} + {byte2!r} -> {merged!r} ('{decoded}')")
        except UnicodeDecodeError:
            print(f"{i+1}. {byte1!r} + {byte2!r} -> {merged!r}")
    
    print("\nLast 10 vocabulary entries:")
    for token_id in sorted(vocab.keys())[-10:]:
        try:
            decoded = vocab[token_id].decode('utf-8')
            print(f"{token_id}: {vocab[token_id]!r} ('{decoded}')")
        except UnicodeDecodeError:
            print(f"{token_id}: {vocab[token_id]!r}")

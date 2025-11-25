"""
BPE Tokenizer Training Module
Implements byte-level Byte-Pair Encoding with parallel pre-tokenization
"""

import os
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

import regex


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    
    Ensures chunk boundaries occur at the beginning of a special token,
    so we never merge across document boundaries.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(chunk_text: str, pattern: regex.Pattern, special_tokens: list[str] | None) -> dict:
    """
    Pre-tokenize a chunk of text and return word frequencies.
    
    Args:
        chunk_text: Text chunk to pre-tokenize
        pattern: Compiled regex pattern for pre-tokenization
        special_tokens: List of special tokens to split on
    
    Returns:
        dict[tuple[bytes, ...], int]: Word frequency table
    """
    word_frequencies = {}
    
    # Split on special tokens BEFORE pre-tokenization
    if special_tokens:
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        text_chunks = [chunk for chunk in re.split(split_pattern, chunk_text) if chunk]
    else:
        text_chunks = [chunk_text]
    
    # Pre-tokenize each chunk separately
    for text in text_chunks:
        for match in pattern.finditer(text):
            word_bytes = match.group().encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in word_bytes)
            word_frequencies[token_tuple] = word_frequencies.get(token_tuple, 0) + 1
    
    return word_frequencies


def pretokenize_worker(args):
    """
    Worker function for parallel pre-tokenization.
    
    Args:
        args: Tuple of (file_path, start_offset, end_offset, pattern_str, special_tokens)
    
    Returns:
        dict[tuple[bytes, ...], int]: Word frequency table for this chunk
    """
    file_path, start, end, pattern_str, special_tokens = args
    
    # Compile pattern in worker process
    pattern = regex.compile(pattern_str)
    
    # Read chunk
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
    
    return pretokenize_chunk(chunk_text, pattern, special_tokens)


def merge_frequency_dicts(dict_list: list[dict]) -> dict:
    """
    Merge multiple frequency dictionaries.
    
    Args:
        dict_list: List of frequency dictionaries
    
    Returns:
        Combined frequency dictionary
    """
    merged = {}
    for d in dict_list:
        for key, count in d.items():
            merged[key] = merged.get(key, 0) + count
    return merged


def get_pretokenized_word_frequencies_parallel(
    input_path: str | os.PathLike,
    pattern: regex.Pattern,
    special_tokens: list[str] | None,
    num_processes: int | None = None,
) -> dict:
    """
    Pre-tokenize corpus in parallel and return frequency table.
    
    Args:
        input_path: Path to corpus file
        pattern: Compiled regex pattern for pre-tokenization
        special_tokens: List of special tokens
        num_processes: Number of parallel processes (defaults to CPU count)
    
    Returns:
        dict[tuple[bytes, ...], int]: Word frequency table
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    # Find chunk boundaries at special token locations
    with open(input_path, 'rb') as f:
        if special_tokens:
            # Use first special token for chunking
            boundaries = find_chunk_boundaries(
                f, 
                num_processes, 
                special_tokens[0].encode('utf-8')
            )
        else:
            # If no special tokens, just split evenly
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            chunk_size = file_size // num_processes
            boundaries = [i * chunk_size for i in range(num_processes + 1)]
            boundaries[-1] = file_size
    
    # Prepare worker arguments
    pattern_str = pattern.pattern
    worker_args = [
        (input_path, start, end, pattern_str, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        chunk_frequencies = pool.map(pretokenize_worker, worker_args)
    
    # Merge results
    return merge_frequency_dicts(chunk_frequencies)


def get_pair_frequencies(word_frequencies: dict) -> dict:
    """
    Count all consecutive byte pairs across all pre-tokenized words.
    
    Args:
        word_frequencies: dict[tuple[bytes, ...], int]
    
    Returns:
        dict[(bytes, bytes), int]: Pair counts weighted by word frequency
    """
    pair_counts = defaultdict(int)
    
    for token_tuple, word_freq in word_frequencies.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] += word_freq
    
    return pair_counts


def merge_pair_in_word(token_tuple: tuple, pair_to_merge: tuple) -> tuple:
    """
    Merge all occurrences of a specific pair within a single word.
    
    Args:
        token_tuple: tuple[bytes, ...] representing a word
        pair_to_merge: (bytes, bytes)
    
    Returns:
        tuple[bytes, ...] with pair merged
    """
    if len(token_tuple) < 2:
        return token_tuple
    
    # Quick check if pair exists
    pair_found = False
    for i in range(len(token_tuple) - 1):
        if token_tuple[i] == pair_to_merge[0] and token_tuple[i + 1] == pair_to_merge[1]:
            pair_found = True
            break
    
    if not pair_found:
        return token_tuple
    
    merged_tokens = []
    i = 0
    
    while i < len(token_tuple):
        if (i < len(token_tuple) - 1 and 
            token_tuple[i] == pair_to_merge[0] and 
            token_tuple[i + 1] == pair_to_merge[1]):
            # Merge
            merged_bytes = pair_to_merge[0] + pair_to_merge[1]
            merged_tokens.append(merged_bytes)
            i += 2
        else:
            merged_tokens.append(token_tuple[i])
            i += 1
    
    return tuple(merged_tokens)


def merge_pair_everywhere_with_cache(
    word_frequencies: dict,
    pair_to_merge: tuple,
    pair_counts: dict,
) -> dict:
    """
    Apply a merge operation and UPDATE pair_counts cache efficiently.
    
    This is the key optimization: instead of recounting all pairs after each merge,
    we only update the pairs that are affected by this specific merge.
    
    Args:
        word_frequencies: Current word frequency table
        pair_to_merge: (bytes, bytes) pair to merge
        pair_counts: Pair frequency cache (modified in-place)
    
    Returns:
        new_word_frequencies: Updated frequency table
    """
    new_word_frequencies = {}
    
    for token_tuple, freq in word_frequencies.items():
        merged_tuple = merge_pair_in_word(token_tuple, pair_to_merge)
        
        if merged_tuple != token_tuple:
            # Word was changed - update pair counts
            
            # Remove old pair counts for this word
            for i in range(len(token_tuple) - 1):
                old_pair = (token_tuple[i], token_tuple[i + 1])
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
            
            # Add new pair counts for merged word
            for i in range(len(merged_tuple) - 1):
                new_pair = (merged_tuple[i], merged_tuple[i + 1])
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq
        
        # Add to new frequency table
        new_word_frequencies[merged_tuple] = new_word_frequencies.get(merged_tuple, 0) + freq
    
    return new_word_frequencies


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.
    
    Args:
        input_path: Path to training text file
        vocab_size: Maximum vocabulary size (including 256 bytes + merges + special tokens)
        special_tokens: List of special tokens (e.g., ["<|endoftext|>"])
    
    Returns:
        vocab: dict[int, bytes] - mapping from token ID to token bytes
        merges: list[tuple[bytes, bytes]] - ordered list of merge operations
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
    
    # Compile the GPT-2 regex pattern for pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenize_pattern = regex.compile(PAT)
    
    # Pre-tokenize corpus in parallel
    word_frequencies = get_pretokenized_word_frequencies_parallel(
        input_path,
        pretokenize_pattern,
        special_tokens,
    )
    
    # Calculate number of merges to perform
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []
    
    # Initial pair counting (only done once!)
    pair_counts = get_pair_frequencies(word_frequencies)
    
    merges = []
    
    for merge_iteration in range(num_merges):
        # Find most frequent pair (with tie-breaking)
        if not pair_counts:
            break
        
        # Get max frequency and choose lexicographically greatest pair
        max_freq = max(pair_counts.values())
        best_pair = max(pair for pair, freq in pair_counts.items() if freq == max_freq)
        
        # Merge this pair throughout the corpus (with cache update)
        word_frequencies = merge_pair_everywhere_with_cache(
            word_frequencies,
            best_pair,
            pair_counts,  # Updated in-place
        )
        
        # Add merged token to vocabulary
        merged_token_bytes = best_pair[0] + best_pair[1]
        vocab[next_token_id] = merged_token_bytes
        next_token_id += 1
        
        # Record this merge operation
        merges.append(best_pair)
    
    return vocab, merges
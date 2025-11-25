import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

import regex as re

# GPT-2 pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
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


def process_chunk(args):
    """
    Process a chunk of the file and return pre-token counts.
    This function is designed to be called in parallel.
    """
    file_path, start, end, special_tokens = args

    # Read the chunk
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')

    # Remove special tokens from the training text entirely
    # The spec says "If these special tokens occur in the input_path, they are treated as any other string"
    # But the test shows they should not appear in the merged tokens
    for special in special_tokens:
        chunk = chunk.replace(special, '')

    # Count pre-tokens in this chunk
    pretoken_counts = defaultdict(int)

    # Use finditer to avoid storing pre-tokenized words
    for match in re.finditer(PAT, chunk):
        pretoken = match.group()
        # Convert to tuple of individual byte objects
        pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode('utf-8'))
        if pretoken_bytes:
            pretoken_counts[pretoken_bytes] += 1

    return pretoken_counts


def get_pair_counts(pretoken_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Count all adjacent pairs of bytes in the pre-tokens.
    """
    pair_counts = defaultdict(int)

    for pretoken, count in pretoken_counts.items():
        if len(pretoken) < 2:
            continue
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i+1])
            pair_counts[pair] += count

    return pair_counts


def merge_pair(pretoken_counts: dict[tuple[bytes], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes], int]:
    """
    Merge all occurrences of the given pair in all pre-tokens.
    Only process pretokens that actually contain both elements of the pair.
    """
    new_pretoken_counts = defaultdict(int)
    a, b = pair

    for pretoken, count in pretoken_counts.items():
        # Quick check: if the pretoken doesn't contain both a and b, skip
        if a not in pretoken or b not in pretoken:
            new_pretoken_counts[pretoken] += count
            continue

        # Merge adjacent occurrences of (a, b) in the pretoken
        new_pretoken = []
        i = 0
        while i < len(pretoken):
            # Check if we can merge at this position
            if i < len(pretoken) - 1 and pretoken[i] == a and pretoken[i+1] == b:
                # Merge the pair
                new_pretoken.append(a + b)
                i += 2
            else:
                # Keep the token as is
                new_pretoken.append(pretoken[i])
                i += 1

        new_pretoken_tuple = tuple(new_pretoken)
        new_pretoken_counts[new_pretoken_tuple] += count

    return dict(new_pretoken_counts)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: Maximum final vocabulary size (including initial bytes, merges, and special tokens).
        special_tokens: List of strings to add to the vocabulary.

    Returns:
        vocab: Mapping from token ID to token bytes.
        merges: List of BPE merges in order of creation.
    """
    # Initialize vocabulary with 256 byte tokens
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    # Add special tokens to vocabulary
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode('utf-8')
        next_token_id += 1

    # Calculate number of merges needed
    num_merges = vocab_size - len(vocab)

    if num_merges <= 0:
        return vocab, []

    # Use multiprocessing to count pre-tokens
    num_processes = cpu_count()

    # Check if <|endoftext|> is in special tokens for efficient chunking
    if '<|endoftext|>' in special_tokens:
        split_token = b'<|endoftext|>'
    else:
        split_token = b'\n\n'  # Fall back to paragraph breaks

    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    # Process chunks in parallel
    chunk_args = [
        (input_path, boundaries[i], boundaries[i+1], special_tokens)
        for i in range(len(boundaries) - 1)
    ]

    with Pool(num_processes) as pool:
        chunk_results = pool.map(process_chunk, chunk_args)

    # Merge counts from all chunks
    pretoken_counts = defaultdict(int)
    for chunk_counts in chunk_results:
        for pretoken, count in chunk_counts.items():
            pretoken_counts[pretoken] += count

    # Perform BPE merges
    merges = []

    for merge_idx in range(num_merges):
        # Get pair counts
        pair_counts = get_pair_counts(pretoken_counts)

        if not pair_counts:
            break

        # Find the most frequent pair (break ties lexicographically)
        max_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        pair = max_pair[0]

        # Add merge to list
        merges.append(pair)

        # Add merged token to vocabulary
        merged_token = pair[0] + pair[1]
        vocab[next_token_id] = merged_token
        next_token_id += 1

        # Merge this pair in all pre-tokens
        pretoken_counts = merge_pair(pretoken_counts, pair)

    return vocab, merges

import os
import regex
import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO, List, Tuple, Dict, Optional

# GPT-2 pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    def __init__(self, vocab=None, merges=None):
        self.vocab = vocab if vocab else {}
        self.merges = merges if merges else []
        self.inverse_vocab = {v: k for k, v in self.vocab.items()} if self.vocab else {}
        self.cache = {}  # Cache for encoding words
        self.pattern = regex.compile(PAT)

    def save(self, path_prefix: str):
        """Save vocab and merges to disk."""
        # Save vocab (convert bytes keys to string representation for JSON)
        # In a real scenario, you might want a more robust serialization for raw bytes
        # Here we assume mostly utf-8 compatible or we use a simple mapping.
        # For simplicity in this assignment, we'll save merges and rebuild vocab or save raw bytes.
        
        # Saving merges is the most important part
        with open(f"{path_prefix}_merges.json", "w") as f:
            # Store merges as list of [byte_str_1, byte_str_2]
            # We encode bytes to latin1 to preserve 1-to-1 mapping for storage if needed,
            # or just keep them as list of ints.
            merges_serializable = [[list(p[0]), list(p[1])] for p in self.merges]
            json.dump(merges_serializable, f)
            
        # We can also save the vocab dict directly
        vocab_serializable = {k: list(v) for k, v in self.vocab.items()}
        with open(f"{path_prefix}_vocab.json", "w") as f:
            json.dump(vocab_serializable, f)

    @classmethod
    def load(cls, path_prefix: str):
        """Load vocab and merges from disk."""
        with open(f"{path_prefix}_merges.json", "r") as f:
            merges_raw = json.load(f)
        
        merges = [
            (bytes(p[0]), bytes(p[1])) for p in merges_raw
        ]
        
        with open(f"{path_prefix}_vocab.json", "r") as f:
            vocab_raw = json.load(f)
            
        vocab = {int(k): bytes(v) for k, v in vocab_raw.items()}
        
        return cls(vocab, merges)

    def train(self, input_path: str, vocab_size: int, special_tokens: List[str] = None):
        """
        Train the tokenizer using the optimized parallel logic.
        """
        print(f"Training BPE on {input_path}...")
        self.vocab, self.merges = train_bpe(input_path, vocab_size, special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Training complete. Vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        """
        if not self.vocab:
            raise ValueError("Tokenizer is not trained. Call train() or load() first.")

        bpe_tokens = []
        # Pre-tokenize
        matches = self.pattern.findall(text)
        
        for token_text in matches:
            # Convert to bytes
            token_bytes = token_text.encode("utf-8")
            
            # Use cache if available
            if token_bytes in self.cache:
                bpe_tokens.extend(self.cache[token_bytes])
                continue

            # Initial state: list of individual bytes
            ids = [self.inverse_vocab[bytes([b])] for b in token_bytes]
            
            # Apply merges greedily
            while len(ids) >= 2:
                stats = get_stats(ids)
                # Find the lowest index merge that is available
                # We need to find the pair in stats that appears earliest in self.merges
                
                # This part can be slow if not optimized. 
                # Standard approach: iterate through merges and apply.
                # Optimized approach: find the best pair in current stats.
                
                # Let's verify which pair in 'stats' has the lowest index in 'self.merges'
                # Optimization: Map pair -> priority (index in merges)
                # This should be pre-calculated ideally, but for now:
                
                best_pair = min(stats.keys(), key=lambda p: self.get_merge_priority(p))
                
                # If the best pair is not in our merges list, we are done
                if self.get_merge_priority(best_pair) == float('inf'):
                    break
                
                # Apply the merge
                ids = merge(ids, best_pair, self.inverse_vocab[self.vocab[ids[0]] + self.vocab[ids[1]] if ids[0] in self.vocab and ids[1] in self.vocab else b""])
                # Wait, calculating the new ID is tricky if we don't track the byte sequence of the ID.
                # Actually, standard BPE encode usually works on tokens, not just IDs.
                # Let's stick to the byte-sequence merge approach for correctness in this context.
                pass 
                
            # Re-implementation of Encode to be strictly consistent with the assignment structure
            # The assignment usually implies: 
            # 1. Start with tuple of bytes: (b'H', b'e', b'l', b'l', b'o')
            # 2. Apply merges
            # 3. Map final chunks to IDs
            
            word = tuple(bytes([b]) for b in token_bytes)
            
            for i in range(len(self.merges)):
                pair = self.merges[i]
                word = merge_pair_in_word(word, pair)
            
            # Map valid tokens to IDs
            word_ids = [self.inverse_vocab[token] for token in word]
            self.cache[token_bytes] = word_ids
            bpe_tokens.extend(word_ids)

        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to string.
        """
        res = b""
        for t in tokens:
            res += self.vocab[t]
        return res.decode("utf-8", errors="replace")

    def get_merge_priority(self, pair):
        # We'll create a lookup for performance on first call
        if not hasattr(self, '_merge_priorities'):
            self._merge_priorities = {pair: i for i, pair in enumerate(self.merges)}
        return self._merge_priorities.get(pair, float('inf'))


# --- Helper Functions for Encoding ---
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# --- The Optimized Training Logic (From bpe_optimized.py) ---

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> List[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

def pretokenize_worker(args):
    file_path, start, end, pattern_str, special_tokens = args
    pattern = regex.compile(pattern_str)
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
    
    word_frequencies = {}
    if special_tokens:
        split_pattern = "|".join(regex.escape(token) for token in special_tokens)
        text_chunks = [c for c in regex.split(split_pattern, chunk_text) if c]
    else:
        text_chunks = [chunk_text]
        
    for text in text_chunks:
        for match in pattern.finditer(text):
            word_bytes = match.group().encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in word_bytes)
            word_frequencies[token_tuple] = word_frequencies.get(token_tuple, 0) + 1
    return word_frequencies

def merge_pair_in_word(token_tuple: tuple, pair_to_merge: tuple) -> tuple:
    if len(token_tuple) < 2: return token_tuple
    # Fast check
    # Note: iterating python tuples is fast, but this can be optimized further
    # For this assignment, the logic is the priority.
    
    # Python implementation of merge
    new_token = []
    i = 0
    plen = len(token_tuple)
    p0, p1 = pair_to_merge
    
    while i < plen:
        if i < plen - 1 and token_tuple[i] == p0 and token_tuple[i+1] == p1:
            new_token.append(p0 + p1)
            i += 2
        else:
            new_token.append(token_tuple[i])
            i += 1
    return tuple(new_token)

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    if special_tokens is None: special_tokens = []
    
    # 1. Init Vocab
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    for st in special_tokens:
        vocab[next_token_id] = st.encode("utf-8")
        next_token_id += 1
        
    # 2. Parallel Pre-tokenize
    num_processes = cpu_count()
    split_token = special_tokens[0].encode('utf-8') if special_tokens else b'\n'
    
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
        
    worker_args = [
        (input_path, boundaries[i], boundaries[i+1], PAT, special_tokens)
        for i in range(len(boundaries) - 1)
    ]
    
    with Pool(num_processes) as pool:
        results = pool.map(pretokenize_worker, worker_args)
        
    word_frequencies = {}
    for res in results:
        for w, freq in res.items():
            word_frequencies[w] = word_frequencies.get(w, 0) + freq
            
    # 3. Train Loop
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0: return vocab, []
    
    # Initial Pair Counts
    pair_counts = defaultdict(int)
    for word, freq in word_frequencies.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i+1])] += freq
            
    merges = []
    
    for i in range(num_merges):
        if not pair_counts: break
        
        # Find best pair (max freq, break ties with lexicographical order)
        # Note: In python max((freq, pair)) works if pair is comparable
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        
        merges.append(best_pair)
        vocab[next_token_id] = best_pair[0] + best_pair[1]
        next_token_id += 1
        
        if (i+1) % 100 == 0:
            print(f"Merge {i+1}/{num_merges}: {best_pair}")
            
        # Update word frequencies and pair counts efficiently
        # We iterate over a copy of items because we modify the dictionary
        words_to_update = []
        for word, freq in word_frequencies.items():
            # Optimization: check if pair is in word before trying to merge
            # This is O(L) where L is word length
            if any(word[k] == best_pair[0] and word[k+1] == best_pair[1] for k in range(len(word)-1)):
                words_to_update.append((word, freq))
                
        for word, freq in words_to_update:
            # 1. Remove old pairs from counts
            for k in range(len(word) - 1):
                pair_counts[(word[k], word[k+1])] -= freq
                if pair_counts[(word[k], word[k+1])] == 0:
                    del pair_counts[(word[k], word[k+1])]
            
            # 2. Update word
            new_word = merge_pair_in_word(word, best_pair)
            del word_frequencies[word]
            word_frequencies[new_word] = word_frequencies.get(new_word, 0) + freq
            
            # 3. Add new pairs to counts
            for k in range(len(new_word) - 1):
                pair_counts[(new_word[k], new_word[k+1])] += freq
                
    return vocab, merges
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import regex as re  # Using 'regex' for \p{L} support
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.transformers.transformers import Embedding, Linear

# GPT-2 pre-tokenization pattern provided in the assignment
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        # Inverse vocab for encoding: bytes -> int
        self.vocab_inv = {v: k for k, v in vocab.items()}
        
        self.merges = merges
        # Rank map for efficient merge lookup: pair -> rank (lower index = higher priority)
        self.merges_ranks = {pair: i for i, pair in enumerate(merges)}
        
        self.special_tokens = special_tokens if special_tokens else []
        
        # Compile the pre-tokenization pattern
        self.pat = re.compile(PAT)
        
        # specific regex for special tokens
        self.special_token_pattern = None
        if self.special_tokens:
            # Sort by length descending to match longest tokens first (e.g. <|end|><|end|> vs <|end|>)
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            # Escape them for regex safety
            escaped_special = [re.escape(s) for s in sorted_special]
            # Create a capturing group so re.split includes the delimiters
            self.special_token_pattern = re.compile(f"({'|'.join(escaped_special)})")


    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "r") as f:
            vocab_raw = json.load(f)
        
        # Reconstruct vocab: {int_id: bytes_token}
        # JSON keys are always strings, so we cast k to int.
        # JSON values are strings, so we encode v to bytes.
        vocab = {
            int(k): v.encode("utf-8") if isinstance(v, str) else v 
            for k, v in vocab_raw.items()
        }

        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                # Handle standard merges file format (e.g. GPT-2) which may have comments
                if line.startswith("#") or not line.strip():
                    continue
                    
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: list[bytes]) -> list[bytes]:
        """Apply BPE merges to a list of bytes (representing a pre-token)."""
        while len(token_bytes) > 1:
            # Find the pair with the lowest rank (earliest in merges list)
            min_rank = float("inf")
            min_pair_idx = -1
            
            for i in range(len(token_bytes) - 1):
                pair = (token_bytes[i], token_bytes[i + 1])
                if pair in self.merges_ranks:
                    rank = self.merges_ranks[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_pair_idx = i
            
            if min_pair_idx == -1:
                break  # No more merges possible
            
            # Merge the best pair
            pair_to_merge = (token_bytes[min_pair_idx], token_bytes[min_pair_idx + 1])
            new_token = pair_to_merge[0] + pair_to_merge[1]
            
            # Replace in the list
            token_bytes[min_pair_idx] = new_token
            del token_bytes[min_pair_idx + 1]
            
        return token_bytes

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        ids = []
        
        # Step 1: Split by special tokens if they exist
        if self.special_token_pattern:
            parts = self.special_token_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
                
            # If this part is a special token
            if part in self.special_tokens:
                # Special tokens are stored as bytes in the vocab
                part_bytes = part.encode("utf-8")
                if part_bytes in self.vocab_inv:
                    ids.append(self.vocab_inv[part_bytes])
                continue

            # Step 2: Pre-tokenize using GPT-2 pattern
            # finditer is generally efficient, but findall is fine for chunks
            pre_tokens = self.pat.findall(part)
            
            for pre_token in pre_tokens:
                # Step 3: Convert to bytes
                pre_token_bytes = [bytes([b]) for b in pre_token.encode("utf-8")]
                
                # Step 4: Apply BPE merges
                merged_tokens = self._bpe(pre_token_bytes)
                
                # Step 5: Map to IDs
                for token in merged_tokens:
                    if token in self.vocab_inv:
                        ids.append(self.vocab_inv[token])
                    else:
                        # Fallback for unknown bytes (should generally be covered by byte-level vocab)
                        pass
                        
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Lazily encode an iterable of strings."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        tokens_bytes = []
        for i in ids:
            if i in self.vocab:
                tokens_bytes.append(self.vocab[i])
        
        # Join all bytes and decode
        full_bytes = b"".join(tokens_bytes)
        # Use errors='replace' for the Unicode replacement character U+FFFD
        return full_bytes.decode("utf-8", errors="replace")


# Detailed Project Structure

**Root:** `/Users/lilysu/Documents/git/assignment1-basics`

## Python Files

- **cs336_basics/__init__.py** (84 bytes)

- **cs336_basics/adapters.py** (25538 bytes)
  - Given the weights of a Linear layer, compute the transformation of a batched input.

- **cs336_basics/bpe.py** (7396 bytes)
  - '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

- **cs336_basics/bpe_optimized.py** (11987 bytes)
  - BPE Tokenizer Training Module

- **cs336_basics/bpe_train_fixed.py** (10581 bytes)
  - Pre-tokenizes text using regex pattern and returns frequency table.

- **cs336_basics/pretokenization_example.py** (2237 bytes)
  - Chunk the file into parts that can be counted independently.

- **cs336_basics/profile_bpe.py** (961 bytes)
  - Profile BPE training to identify bottlenecks.

- **cs336_basics/train_analysis/compare_tokenizers.py** (7653 bytes)
  - Compare BPE tokenizers trained on TinyStories vs OpenWebText

- **cs336_basics/train_analysis/profile_training.py** (4336 bytes)
  - Profile BPE training to identify bottlenecks

- **cs336_basics/train_analysis/train_openwebtext.py** (5314 bytes)
  - Train BPE tokenizer on OpenWebText dataset

- **cs336_basics/train_analysis/train_tinystories.py** (4201 bytes)
  - Train BPE tokenizer on TinyStories dataset

- **cs336_basics/train_tinystories_bpe.py** (3715 bytes)
  - Train a BPE tokenizer on the TinyStories or OpenWebText dataset.

- **cs336_basics/verify_bpe.py** (8994 bytes)
  - Quick verification script for BPE implementation

- **tests/__init__.py** (0 bytes)

- **tests/adapters.py** (25548 bytes)
  - Given the weights of a Linear layer, compute the transformation of a batched input.

- **tests/common.py** (2353 bytes)
  - Returns a mapping between every possible byte (an integer from 0 to 255) to a

- **tests/conftest.py** (9160 bytes)
  - Snapshot testing utility for NumPy arrays using .npz format.

- **tests/test_data.py** (2823 bytes)

- **tests/test_model.py** (6352 bytes)

- **tests/test_nn_utils.py** (3098 bytes)

- **tests/test_optimizer.py** (2780 bytes)
  - Our reference implementation yields slightly different results than the

- **tests/test_serialization.py** (3878 bytes)

- **tests/test_tokenizer.py** (16447 bytes)
  - We expect this test to fail, since Tokenizer.encode is not expected to be memory efficient.

- **tests/test_train_bpe.py** (3246 bytes)
  - Ensure that BPE training is relatively efficient by measuring training

- **tree.py** (1320 bytes)
  - Simple one-line project tree visualizer

- **visualize_structure.py** (7611 bytes)
  - Visualize project folder structure in markdown format

## Documentation Files

- **.pytest_cache/README.md** (302 bytes)
- **CHANGELOG.md** (6883 bytes)
- **PROJECT_STRUCTURE.md** (3915 bytes)
- **README.md** (2252 bytes)
- **cs336_basics/PROJECT_STRUCTURE.md** (1015 bytes)
- **cs336_basics/PROJECT_STRUCTURE_DETAILED.md** (1799 bytes)
- **cs336_basics/train_analysis/PROJECT_STRUCTURE.md** (428 bytes)
- **cs336_basics/train_analysis/PROJECT_STRUCTURE_DETAILED.md** (759 bytes)
- **cs336_basics/conceptual_answers.txt** (1901 bytes)
- **data/TinyStoriesV2-GPT4-train.txt** (2227753162 bytes)
- **data/TinyStoriesV2-GPT4-valid.txt** (22502601 bytes)
- **data/owt_train.txt** (11920511059 bytes)
- **data/owt_valid.txt** (289998753 bytes)
- **tests/fixtures/address.txt** (1468 bytes)
- **tests/fixtures/german.txt** (594 bytes)
- **tests/fixtures/gpt2_merges.txt** (456304 bytes)
- **tests/fixtures/special_token_double_newlines_non_whitespace.txt** (23 bytes)
- **tests/fixtures/special_token_trailing_newlines.txt** (15 bytes)
- **tests/fixtures/tinystories_sample.txt** (3794 bytes)
- **tests/fixtures/tinystories_sample_5M.txt** (5242880 bytes)
- **tests/fixtures/train-bpe-reference-merges.txt** (1271 bytes)

## Data Files

- **data/TinyStoriesV2-GPT4-train.txt** (2124.6MB)
- **data/TinyStoriesV2-GPT4-valid.txt** (21.5MB)
- **data/owt_train.txt** (11368.3MB)
- **data/owt_valid.txt** (276.6MB)

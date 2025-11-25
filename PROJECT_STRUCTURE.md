# Project Structure: assignment1-basics

**Root:** `/Users/lilysu/Documents/git/assignment1-basics`

```
assignment1-basics/
├── cs336_basics/
│   ├── train_analysis/
│   │   ├── compare_tokenizers.py (7.5KB)
│   │   ├── profile_training.py (4.2KB)
│   │   ├── PROJECT_STRUCTURE.md (428B)
│   │   ├── PROJECT_STRUCTURE_DETAILED.md (759B)
│   │   ├── train_openwebtext.py (5.2KB)
│   │   └── train_tinystories.py (4.1KB)
│   ├── __init__.py (84B)
│   ├── adapters.py (24.9KB)
│   ├── bpe.ipynb (48.1KB)
│   ├── bpe.py (7.2KB)
│   ├── bpe_optimized.py (11.7KB)
│   ├── bpe_train_fixed.py (10.3KB)
│   ├── conceptual_answers.txt (1.9KB)
│   ├── pretokenization_example.py (2.2KB)
│   ├── profile_bpe.py (961B)
│   ├── PROJECT_STRUCTURE.md (1015B)
│   ├── PROJECT_STRUCTURE_DETAILED.md (1.8KB)
│   ├── train_bpe.ipynb (9.6KB)
│   ├── train_bpe_sandbox.ipynb (120.6KB)
│   ├── train_tinystories_bpe.py (3.6KB)
│   └── verify_bpe.py (8.8KB)
├── data/
│   ├── owt_train.txt (11.1GB)
│   ├── owt_valid.txt (276.6MB)
│   ├── TinyStoriesV2-GPT4-train.txt (2.1GB)
│   └── TinyStoriesV2-GPT4-valid.txt (21.5MB)
├── output/
│   ├── tinystories_merges.pkl (129.3KB)
│   └── tinystories_vocab.pkl (114.4KB)
├── tests/
│   ├── _snapshots/
│   │   ├── test_4d_scaled_dot_product_attention.npz (12.3KB)
│   │   ├── test_adamw.npz (288B)
│   │   ├── test_embedding.npz (12.3KB)
│   │   ├── test_linear.npz (24.3KB)
│   │   ├── test_multihead_self_attention.npz (12.3KB)
│   │   ├── test_multihead_self_attention_with_rope.npz (12.3KB)
│   │   ├── test_positionwise_feedforward.npz (12.3KB)
│   │   ├── test_rmsnorm.npz (12.3KB)
│   │   ├── test_rope.npz (12.3KB)
│   │   ├── test_scaled_dot_product_attention.npz (12.3KB)
│   │   ├── test_swiglu.npz (12.3KB)
│   │   ├── test_train_bpe_special_tokens.pkl (17.4KB)
│   │   ├── test_transformer_block.npz (12.3KB)
│   │   ├── test_transformer_lm.npz (1.8MB)
│   │   └── test_transformer_lm_truncated_input.npz (937.8KB)
│   ├── fixtures/
│   │   ├── ts_tests/
│   │   │   ├── model.pt (5.4MB)
│   │   │   └── model_config.json (262B)
│   │   ├── address.txt (1.4KB)
│   │   ├── corpus.en (129.9KB)
│   │   ├── german.txt (594B)
│   │   ├── gpt2_merges.txt (445.6KB)
│   │   ├── gpt2_vocab.json (1017.9KB)
│   │   ├── special_token_double_newlines_non_whitespace.txt (23B)
│   │   ├── special_token_trailing_newlines.txt (15B)
│   │   ├── tinystories_sample.txt (3.7KB)
│   │   ├── tinystories_sample_5M.txt (5.0MB)
│   │   ├── train-bpe-reference-merges.txt (1.2KB)
│   │   └── train-bpe-reference-vocab.json (7.4KB)
│   ├── __init__.py (0B)
│   ├── adapters.py (24.9KB)
│   ├── common.py (2.3KB)
│   ├── conftest.py (8.9KB)
│   ├── test_data.py (2.8KB)
│   ├── test_model.py (6.2KB)
│   ├── test_nn_utils.py (3.0KB)
│   ├── test_optimizer.py (2.7KB)
│   ├── test_serialization.py (3.8KB)
│   ├── test_tokenizer.py (16.1KB)
│   └── test_train_bpe.py (3.2KB)
├── CHANGELOG.md (6.7KB)
├── cs336_spring2025_assignment1_basics.pdf (409.7KB)
├── LICENSE (1.0KB)
├── make_submission.sh (844B)
├── pyproject.toml (1.2KB)
├── README.md (2.2KB)
├── tree.py (1.3KB)
├── uv.lock (158.9KB)
└── visualize_structure.py (7.4KB)
```

**Statistics:**
- Total directories: 8
- Total files: 75
# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Usage examples:

```uv venv```

```source .venv/bin/activate```

#### Train on TinyStories (default, vocab size 10000)
```python cs336_basics/train_tinystories_bpe.py```

#### Train on OpenWebText with vocab size 32000
```python cs336_basics/train_tinystories_bpe.py --dataset openwebtext```

#### Or use the shorthand
```python cs336_basics/train_tinystories_bpe.py --dataset owt```

#### Custom vocab size for OpenWebText
```python cs336_basics/train_tinystories_bpe.py --dataset owt --vocab-size 50000```

#### Custom output prefix
```python cs336_basics/train_tinystories_bpe.py --dataset owt --output-prefix owt_32k```


#### Tokenizer Test
```uv run pytest tests/test_tokenizer.py```

#### Linear Test
```uv run pytest -k test_linear```

#### Embedding Test
```uv run pytest -k test_embedding```

#### RMS Norm Test
```uv run pytest -k test_rmsnorm```

#### Swiglu Test
```uv run pytest -k test_swiglu```

#### Rope Test
```uv run pytest -k test_rope```

#### Softmax Test
```uv run pytest -k test_softmax_matches_pytorch```

#### Scaled Dot Attention with Boolean Mask Test
```uv run pytest -k test_4d_scaled_dot_product_attention```

#### Causal Masking With Rope Test
```uv run pytest -k test_multihead_self_attention_with_rope```

#### Transformer Block Test
```uv run pytest -k test_transformer_block```

#### Implement the TransformerLM Test
```uv run pytest -k test_transformer_lm```

#### Cross Entropy Test
```uv run pytest -k test_cross_entropy```

#### AdamW Test
```uv run pytest -k test_adamw```

#### Cosine Learning Rate Test
```uv run pytest -k test_get_lr_cosine_schedule```

#### Gradient Clipping Test
```uv run pytest -k test_gradient_clipping```

#### Data Loading Get Batch Test
```uv run pytest -k test_get_batch```

#### Checkpointing Test
```uv run pytest -k test_checkpointing```
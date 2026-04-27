# Tiny Transformer LM

Minimal Transformer LM implementation (BPE tokenizer + model + training loop)
for CS336 Assignment 1. Trained on a 32k BPE tokenizer over OpenWebText.

## Setup

Install [`uv`](https://github.com/astral-sh/uv), then:

```sh
uv sync
```

## Data

Pre-tokenized OWT (32k BPE) lives in a private HF dataset. Pull it into
`data/` — `train.py` reads from there directly:

```sh
export HF_TOKEN=hf_xxx   # read token for AndrewK101/cs336-owt-32k-bpe
uv run hf download AndrewK101/cs336-owt-32k-bpe \
    --repo-type dataset --local-dir data
```

After download, `data/` contains:

| file | what |
|---|---|
| `owt_train.npy` | uint16 token ids of the OWT training split (2.73B tokens, 5.1 GB) |
| `owt_valid.npy` | uint16 token ids of the OWT validation split (66M tokens, 127 MB) |
| `owt_vocab_32k.pkl` | `dict[int, bytes]` — special at id 0, bytes 1..256, merges 257..31999 |
| `owt_merges_32k.pkl` | `list[tuple[bytes, bytes]]` in merge order |

The id scheme matches `cs336_basics.tokenizer.Tokenizer`. `train.py` is
already pointed at these paths; no edits needed.

### Reproducing the dataset from raw OWT

If you want to rebuild the cache yourself instead of downloading:

```sh
mkdir -p data && cd data
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_train.txt.gz owt_valid.txt.gz
cd ..
uv run python train_owt_tokenizer.py   # 500 MB sample -> 32k vocab/merges (~2 min)
uv run python scripts/tokenize_owt.py  # encode train+valid via tiktoken (~10 min CPU)
```

## Train

```sh
uv run train.py
```

Hyperparameters (paths, model size, optimizer, logging) are defined as
constants at the top of `train.py` — edit there to change a run.
Checkpoints are saved to `checkpoints/model.pt`.

## Layout

```
cs336_basics/
  bpe.py          # BPE trainer (parallel pretokenize + heap merge loop)
  tokenizer.py    # BPE tokenizer (encode/decode/encode_iterable/from_files)
  model.py        # Transformer LM (Linear, Embedding, RMSNorm, SwiGLU, RoPE, MHA, ...)
  training.py     # AdamW, cosine LR, cross-entropy, grad clip, get_batch, checkpoints
train.py                  # training script
train_owt_tokenizer.py    # 500 MB OWT sample -> 32k BPE
scripts/tokenize_owt.py   # text -> uint16 .npy via tiktoken (parity-checked)
scripts/verify_parallel_bpe.py  # determinism check vs reference impl
```

# Tiny Transformer LM

Minimal Transformer LM implementation (BPE tokenizer + model + training loop).

## Setup

Install [`uv`](https://github.com/astral-sh/uv), then:

```sh
uv sync
```

## Download data

```sh
mkdir -p data && cd data

# TinyStories
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText sample (optional)
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz

cd ..
```

## Train

```sh
uv run train.py
```

First run will train a BPE tokenizer (cached to `data/vocab.pkl`, `data/merges.pkl`)
and tokenize the corpora (cached to `data/train.npy`, `data/valid.npy`).
Checkpoints are saved to `checkpoints/model.pt`.

All hyperparameters (data paths, model size, optimizer, logging) are defined as
constants at the top of `train.py` — edit them there to change a run.

## Layout

```
cs336_basics/
  bpe.py          # BPE trainer
  tokenizer.py    # BPE tokenizer (encode/decode/encode_iterable)
  model.py        # Transformer LM (Linear, Embedding, RMSNorm, SwiGLU, RoPE, MHA, ...)
  training.py     # AdamW, cosine LR, cross-entropy, grad clip, get_batch, checkpoints
train.py          # training script
```

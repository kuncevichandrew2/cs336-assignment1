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

### Common overrides

```sh
uv run train.py \
  --train-text data/owt_train.txt \
  --valid-text data/owt_valid.txt \
  --vocab-size 32000 \
  --d-model 768 --num-layers 6 --num-heads 12 --d-ff 2048 \
  --context-length 512 --batch-size 16 \
  --max-iters 20000 --lr-max 3e-4
```

Resume: add `--resume`. See `train.py` for all flags.

## Layout

```
cs336_basics/
  bpe.py          # BPE trainer
  tokenizer.py    # BPE tokenizer (encode/decode/encode_iterable)
  model.py        # Transformer LM (Linear, Embedding, RMSNorm, SwiGLU, RoPE, MHA, ...)
  training.py     # AdamW, cosine LR, cross-entropy, grad clip, get_batch, checkpoints
train.py          # training script
```

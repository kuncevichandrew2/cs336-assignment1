from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.bpe import train_bpe
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


# ---- config ----
# data
TRAIN_TEXT = "data/TinyStoriesV2-GPT4-train.txt"
VALID_TEXT = "data/TinyStoriesV2-GPT4-valid.txt"
TRAIN_TOKENS = "data/train.npy"
VALID_TOKENS = "data/valid.npy"
VOCAB_PATH = "data/vocab.pkl"
MERGES_PATH = "data/merges.pkl"
VOCAB_SIZE = 10_000
# model
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
ROPE_THETA = 10000.0
# optim
BATCH_SIZE = 32
MAX_ITERS = 5000
LR_MAX = 3e-4
LR_MIN = 3e-5
WARMUP_ITERS = 200
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0
# logging / checkpointing
EVAL_INTERVAL = 500
EVAL_ITERS = 50
LOG_INTERVAL = 50
CKPT_PATH = "checkpoints/model.pt"
RESUME = False
# system
DEVICE: str | None = None
SEED = 42


def tokenize_file(tokenizer: Tokenizer, text_path: Path, out_path: Path) -> np.ndarray:
    """Stream-tokenize a text file into a .npy array of uint16/uint32 token ids."""
    print(f"Tokenizing {text_path} -> {out_path}")
    ids: list[int] = []
    with open(text_path) as f:
        for _id in tokenizer.encode_iterable(f):
            ids.append(_id)
    vocab_size = len(tokenizer.vocab)
    dtype = np.uint16 if vocab_size < 2**16 else np.uint32
    arr = np.array(ids, dtype=dtype)
    np.save(out_path, arr)
    print(f"  wrote {len(arr):,} tokens")
    return arr


def build_or_load_tokenizer() -> Tokenizer:
    vocab_path = Path(VOCAB_PATH)
    merges_path = Path(MERGES_PATH)
    if vocab_path.exists() and merges_path.exists():
        print(f"Loading tokenizer from {vocab_path}, {merges_path}")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
    else:
        print(f"Training BPE on {TRAIN_TEXT} (vocab_size={VOCAB_SIZE})")
        t0 = time.time()
        vocab, merges = train_bpe(
            input_path=TRAIN_TEXT,
            vocab_size=VOCAB_SIZE,
            special_tokens=["<|endoftext|>"],
        )
        print(f"  trained in {time.time() - t0:.1f}s, |V|={len(vocab)}")
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        with open(merges_path, "wb") as f:
            pickle.dump(merges, f)
    return Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])


def load_or_tokenize_dataset(tokenizer: Tokenizer, text_path: str, cache_path: str) -> np.ndarray:
    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading tokens from {cache}")
        return np.load(cache, mmap_mode="r")
    cache.parent.mkdir(parents=True, exist_ok=True)
    return tokenize_file(tokenizer, Path(text_path), cache)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = DEVICE
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"device={device}")

    # ---- tokenizer + data ----
    tokenizer = build_or_load_tokenizer()
    train_ids = load_or_tokenize_dataset(tokenizer, TRAIN_TEXT, TRAIN_TOKENS)
    valid_ids = load_or_tokenize_dataset(tokenizer, VALID_TEXT, VALID_TOKENS)

    # ---- model ----
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M")

    optimizer = AdamW(
        model.parameters(),
        lr=LR_MAX,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    start_iter = 0
    ckpt = Path(CKPT_PATH)
    if RESUME and ckpt.exists():
        start_iter = load_checkpoint(ckpt, model, optimizer)
        print(f"resumed at iter {start_iter}")

    # ---- train loop ----
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for it in range(start_iter, MAX_ITERS):
        lr = get_lr_cosine_schedule(it, LR_MAX, LR_MIN, WARMUP_ITERS, MAX_ITERS)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it % LOG_INTERVAL == 0:
            dt = time.time() - t0
            print(f"iter {it:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s")

        if it > 0 and it % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(EVAL_ITERS):
                    xv, yv = get_batch(valid_ids, BATCH_SIZE, CONTEXT_LENGTH, device)
                    lv = cross_entropy(model(xv).view(-1, logits.size(-1)), yv.view(-1))
                    losses.append(lv.item())
                val = float(np.mean(losses))
            model.train()
            print(f"iter {it:6d} | VAL loss {val:.4f}")
            save_checkpoint(model, optimizer, it, ckpt)

    save_checkpoint(model, optimizer, MAX_ITERS, ckpt)
    print(f"done. checkpoint: {ckpt}")


if __name__ == "__main__":
    main()

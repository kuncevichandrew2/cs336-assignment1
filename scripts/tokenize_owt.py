"""Tokenize OWT train+valid into uint16 .npy using our trained 32k BPE.

Uses tiktoken (C-based, parallel-batch) for speed. Verifies parity vs the
pure-Python `cs336_basics.tokenizer.Tokenizer` on a small sample before
encoding the full corpus.

Output IDs use the SAME scheme as our `Tokenizer`/`Tokenizer.from_files`:
- special <|endoftext|> -> 0
- byte b'\\x00'..b'\\xff' -> 1..256
- merged tokens -> 257..31999
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs336_basics.tokenizer import GPT2_PAT, Tokenizer

VOCAB_PATH = Path("data/owt_vocab_32k.pkl")
MERGES_PATH = Path("data/owt_merges_32k.pkl")
SPECIAL = "<|endoftext|>"
SPECIAL_BYTES = SPECIAL.encode("utf-8")

TRAIN_TXT = Path("data/owt_train.txt")
VALID_TXT = Path("data/owt_valid.txt")
TRAIN_NPY = Path("data/owt_train.npy")
VALID_NPY = Path("data/owt_valid.npy")

# Read in ~256 MB slices, snap each slice to the last <|endoftext|> boundary,
# pass slices to tiktoken.encode_ordinary_batch (parallel via rayon).
SLICE_BYTES = 256 * 1024 * 1024


def build_tiktoken(vocab: dict[int, bytes]) -> tuple[tiktoken.Encoding, int]:
    """Build a tiktoken Encoding mirroring our vocab.

    Our scheme: special at 0, bytes/merges at 1..N-1.
    Tiktoken requires contiguous ranks 0..N-2 (ex-special) and a separate
    special-token id. We map our-id -> tiktoken-rank = our-id - 1, and put
    the special at tiktoken-id = N-1. After encoding we remap back.
    """
    mergeable_ranks: dict[bytes, int] = {}
    n = len(vocab)
    for our_id, tok in vocab.items():
        if tok == SPECIAL_BYTES:
            continue
        mergeable_ranks[tok] = our_id - 1
    tt_special_id = n - 1
    enc = tiktoken.Encoding(
        name="cs336_owt_32k",
        pat_str=GPT2_PAT,
        mergeable_ranks=mergeable_ranks,
        special_tokens={SPECIAL: tt_special_id},
    )
    return enc, tt_special_id


def remap_to_our_ids(tt_ids: np.ndarray, tt_special_id: int) -> np.ndarray:
    """tiktoken rank r -> our id r+1; tiktoken-special -> our id 0."""
    out = (tt_ids + 1).astype(np.uint32)
    out[tt_ids == tt_special_id] = 0
    assert out.max() < 65536
    return out.astype(np.uint16)


def verify_parity(enc: tiktoken.Encoding, tt_special_id: int,
                  vocab: dict[int, bytes],
                  merges: list[tuple[bytes, bytes]],
                  sample: str) -> None:
    """Round-trip a sample through both encoders and assert matching IDs."""
    py_tok = Tokenizer(vocab, merges, special_tokens=[SPECIAL])
    py_ids = py_tok.encode(sample)

    tt_ids = np.asarray(enc.encode(sample, allowed_special={SPECIAL}), dtype=np.int64)
    tt_mapped = remap_to_our_ids(tt_ids, tt_special_id).tolist()

    if py_ids != tt_mapped:
        # locate first divergence
        n = min(len(py_ids), len(tt_mapped))
        for i in range(n):
            if py_ids[i] != tt_mapped[i]:
                ctx_lo = max(0, i - 3)
                raise AssertionError(
                    f"parity mismatch at i={i}: py={py_ids[ctx_lo:i+4]} tt={tt_mapped[ctx_lo:i+4]}"
                )
        raise AssertionError(f"length mismatch: py={len(py_ids)} tt={len(tt_mapped)}")
    print(f"parity OK on {len(sample):,} chars -> {len(py_ids):,} tokens")


def encode_file(enc: tiktoken.Encoding, tt_special_id: int,
                src: Path, dst: Path) -> None:
    print(f"encoding {src} -> {dst}")
    size = src.stat().st_size
    chunks_processed = 0
    out_arrays: list[np.ndarray] = []
    t0 = time.time()
    with open(src, "rb") as f:
        leftover = b""
        while True:
            buf = f.read(SLICE_BYTES)
            if not buf:
                if leftover:
                    text = leftover.decode("utf-8", errors="ignore")
                    tt_ids = np.asarray(
                        enc.encode_ordinary_batch([text], num_threads=os.cpu_count() or 8)[0],
                        dtype=np.int64,
                    )
                    out_arrays.append(remap_to_our_ids(tt_ids, tt_special_id))
                    leftover = b""
                break
            buf = leftover + buf
            # Snap to the last <|endoftext|> boundary so tokenization is stable
            # across slices. (\n boundary would also work since GPT-2 regex
            # never crosses \n, but the special token guarantees that no merge
            # spans the boundary either.)
            cut = buf.rfind(SPECIAL_BYTES)
            if cut == -1:
                # Fallback to last newline if no special token in slice.
                cut = buf.rfind(b"\n")
                if cut == -1:
                    leftover = buf
                    continue
                cut += 1
            else:
                cut += len(SPECIAL_BYTES)
            head, leftover = buf[:cut], buf[cut:]

            # Split head into a few documents per call so encode_ordinary_batch
            # can parallelize.
            text = head.decode("utf-8", errors="ignore")
            docs = text.split(SPECIAL)
            # Rejoin with the special token suffix on each doc except possibly
            # the last (which may or may not end with the special token).
            pieces = []
            for i, d in enumerate(docs):
                if i < len(docs) - 1:
                    pieces.append(d + SPECIAL)
                elif d:
                    pieces.append(d)
            if not pieces:
                continue
            batched = enc.encode_batch(pieces, allowed_special={SPECIAL}, num_threads=os.cpu_count() or 8)
            for sub in batched:
                arr = np.asarray(sub, dtype=np.int64)
                out_arrays.append(remap_to_our_ids(arr, tt_special_id))

            chunks_processed += 1
            done_bytes = f.tell()
            dt = time.time() - t0
            mb = done_bytes / 1024 / 1024
            print(f"  slice {chunks_processed}: {mb:,.0f}/{size/1024/1024:,.0f} MB | {dt:.1f}s | {mb/max(dt,1e-9):.1f} MB/s")

    print(f"concatenating {len(out_arrays)} arrays...")
    arr = np.concatenate(out_arrays)
    print(f"  total tokens: {len(arr):,} ({len(arr)*2/1024/1024/1024:.2f} GB uint16)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr)
    print(f"  wrote {dst} ({dst.stat().st_size/1024/1024:.1f} MB)")


def main() -> None:
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(MERGES_PATH, "rb") as f:
        merges = pickle.load(f)
    print(f"loaded |vocab|={len(vocab)} |merges|={len(merges)}")

    enc, tt_special_id = build_tiktoken(vocab)

    # Parity check on first 256 KB of valid (or train if no valid).
    src = VALID_TXT if VALID_TXT.exists() else TRAIN_TXT
    with open(src, "rb") as f:
        sample = f.read(256 * 1024).decode("utf-8", errors="ignore")
    nl = sample.rfind("\n")
    if nl != -1:
        sample = sample[: nl + 1]
    verify_parity(enc, tt_special_id, vocab, merges, sample)

    if VALID_TXT.exists():
        encode_file(enc, tt_special_id, VALID_TXT, VALID_NPY)
    if TRAIN_TXT.exists():
        encode_file(enc, tt_special_id, TRAIN_TXT, TRAIN_NPY)


if __name__ == "__main__":
    main()

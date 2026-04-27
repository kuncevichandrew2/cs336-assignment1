"""Train a 32k BPE tokenizer on a 500MB sub-sample of OpenWebText.

Reads data/owt_train.txt, takes the first ~500 MB (truncated at the last
newline so we don't split mid-line), writes data/owt_train_sample500mb.txt
if missing, then trains BPE with vocab_size=32000 and special token
"<|endoftext|>". Saves vocab and merges to data/owt_vocab_32k.pkl and
data/owt_merges_32k.pkl.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe

INPUT_PATH = Path("data/owt_train.txt")
SAMPLE_PATH = Path("data/owt_train_sample500mb.txt")
VOCAB_PATH = Path("data/owt_vocab_32k.pkl")
MERGES_PATH = Path("data/owt_merges_32k.pkl")
SAMPLE_BYTES = 500 * 1024 * 1024
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]


def make_sample(src: Path, dst: Path, max_bytes: int) -> None:
    if dst.exists() and dst.stat().st_size >= max_bytes * 0.95:
        print(f"sample already exists: {dst} ({dst.stat().st_size:,} bytes)")
        return
    print(f"creating {max_bytes:,}-byte sample: {src} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f:
        buf = f.read(max_bytes)
    # Truncate at the last newline so we don't end mid-line.
    nl = buf.rfind(b"\n")
    if nl != -1:
        buf = buf[: nl + 1]
    with open(dst, "wb") as f:
        f.write(buf)
    print(f"  wrote {len(buf):,} bytes")


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"missing {INPUT_PATH} — download owt_train.txt.gz first")

    make_sample(INPUT_PATH, SAMPLE_PATH, SAMPLE_BYTES)

    print(f"training BPE: vocab_size={VOCAB_SIZE}, special={SPECIAL_TOKENS}")
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=SAMPLE_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )
    dt = time.time() - t0
    print(f"trained in {dt:.1f}s | |vocab|={len(vocab)} | |merges|={len(merges)}")

    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    with open(MERGES_PATH, "wb") as f:
        pickle.dump(merges, f)
    print(f"saved: {VOCAB_PATH} ({VOCAB_PATH.stat().st_size:,} B), "
          f"{MERGES_PATH} ({MERGES_PATH.stat().st_size:,} B)")


if __name__ == "__main__":
    main()

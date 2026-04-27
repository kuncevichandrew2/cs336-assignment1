"""Verify parallel pretokenization is bitwise-identical to serial.

Trains BPE twice on the same 5 MB sub-sample of data/owt_valid.txt:
once with the serial pretokenizer, once with the parallel one. Asserts
that vocab and merges match exactly. Exits non-zero on mismatch.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Allow importing the package when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics import bpe as bpe_mod
from cs336_basics.bpe import _pretokenize_text, _pretokenize_text_serial, train_bpe

VALID_PATH = Path("data/owt_valid.txt")
SAMPLE_BYTES = 5 * 1024 * 1024
VOCAB_SIZE = 2000
SPECIAL_TOKENS = ["<|endoftext|>"]


def _train_with(pretokenizer, text: str, vocab_size: int, special_tokens: list[str]):
    word_counts = pretokenizer(text, special_tokens)

    vocab: dict[int, bytes] = {}
    for s in special_tokens:
        vocab[len(vocab)] = s.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])

    merges: list[tuple[bytes, bytes]] = []
    words: list[list[bytes]] = []
    word_freqs: list[int] = []
    for w, c in word_counts.items():
        words.append(list(w))
        word_freqs.append(c)

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for wi, toks in enumerate(words):
        freq = word_freqs[wi]
        for i in range(len(toks) - 1):
            pair = (toks[i], toks[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(wi)

    target_merges = vocab_size - len(vocab)
    for _ in range(target_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best] <= 0:
            break
        merges.append(best)
        new_tok = best[0] + best[1]
        vocab[len(vocab)] = new_tok
        affected = list(pair_to_words[best])
        for wi in affected:
            toks = words[wi]
            freq = word_freqs[wi]
            for i in range(len(toks) - 1):
                p = (toks[i], toks[i + 1])
                pair_counts[p] -= freq
                pair_to_words[p].discard(wi)
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                    pair_to_words.pop(p, None)
            new_toks: list[bytes] = []
            i = 0
            while i < len(toks):
                if i < len(toks) - 1 and toks[i] == best[0] and toks[i + 1] == best[1]:
                    new_toks.append(new_tok)
                    i += 2
                else:
                    new_toks.append(toks[i])
                    i += 1
            words[wi] = new_toks
            for i in range(len(new_toks) - 1):
                p = (new_toks[i], new_toks[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(wi)

    return vocab, merges


def main() -> int:
    if not VALID_PATH.exists():
        print(f"missing {VALID_PATH}", file=sys.stderr)
        return 2

    with open(VALID_PATH, "rb") as f:
        buf = f.read(SAMPLE_BYTES)
    nl = buf.rfind(b"\n")
    if nl != -1:
        buf = buf[: nl + 1]
    text = buf.decode("utf-8", errors="ignore")
    print(f"verify corpus: {len(text):,} chars (~{len(buf):,} bytes)")

    t0 = time.time()
    vocab_s, merges_s = _train_with(_pretokenize_text_serial, text, VOCAB_SIZE, SPECIAL_TOKENS)
    t_serial = time.time() - t0
    print(f"serial:   {t_serial:.2f}s | |V|={len(vocab_s)} | |M|={len(merges_s)}")

    t0 = time.time()
    vocab_p, merges_p = _train_with(_pretokenize_text, text, VOCAB_SIZE, SPECIAL_TOKENS)
    t_parallel = time.time() - t0
    print(f"parallel: {t_parallel:.2f}s | |V|={len(vocab_p)} | |M|={len(merges_p)}")

    if vocab_s != vocab_p:
        print("VOCAB MISMATCH (pretokenize)", file=sys.stderr)
        return 1
    if merges_s != merges_p:
        print("MERGES MISMATCH (pretokenize)", file=sys.stderr)
        for i, (a, b) in enumerate(zip(merges_s, merges_p)):
            if a != b:
                print(f"  first diverge at i={i}: serial={a} parallel={b}", file=sys.stderr)
                break
        return 1

    # Now also exercise the optimized train_bpe (parallel + heap merge loop)
    # against a temp file containing the same corpus.
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as tmp:
        tmp.write(text.encode("utf-8"))
        tmp_path = tmp.name
    try:
        t0 = time.time()
        vocab_t, merges_t = train_bpe(
            input_path=tmp_path,
            vocab_size=VOCAB_SIZE,
            special_tokens=SPECIAL_TOKENS,
        )
        t_train = time.time() - t0
    finally:
        os.unlink(tmp_path)
    print(f"train_bpe: {t_train:.2f}s | |V|={len(vocab_t)} | |M|={len(merges_t)}")

    if vocab_t != vocab_s:
        print("VOCAB MISMATCH (train_bpe)", file=sys.stderr)
        return 1
    if merges_t != merges_s:
        print("MERGES MISMATCH (train_bpe vs reference)", file=sys.stderr)
        for i, (a, b) in enumerate(zip(merges_s, merges_t)):
            if a != b:
                print(f"  first diverge at i={i}: ref={a} train_bpe={b}", file=sys.stderr)
                break
        return 1

    print("OK: vocab and merges identical (pretokenize parity + train_bpe parity)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import os
import regex as re
from collections import Counter, defaultdict
from typing import BinaryIO

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(GPT2_PAT)


def _pretokenize_text(text: str, special_tokens: list[str]) -> Counter:
    """Split text on special tokens, then apply GPT-2 pretokenizer to each chunk. Returns Counter of tuple[bytes]."""
    counts: Counter = Counter()
    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]
    for chunk in chunks:
        for match in PAT_RE.finditer(chunk):
            tok = match.group(0).encode("utf-8")
            counts[tuple(bytes([b]) for b in tok)] += 1
    return counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        data = f.read().decode("utf-8", errors="ignore")

    word_counts = _pretokenize_text(data, special_tokens)

    # Initialize vocab: special tokens first, then all bytes 0..255
    vocab: dict[int, bytes] = {}
    for s in special_tokens:
        vocab[len(vocab)] = s.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])

    merges: list[tuple[bytes, bytes]] = []

    # Store each unique word as a list of tokens (bytes) with its frequency
    words: list[list[bytes]] = []
    word_freqs: list[int] = []
    for w, c in word_counts.items():
        words.append(list(w))
        word_freqs.append(c)

    # pair -> total count; pair -> set of word indices containing it
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
        # Find the best pair (max count, tie-broken by lexicographic max of the pair of bytes)
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best] <= 0:
            break
        merges.append(best)
        new_tok = best[0] + best[1]
        vocab[len(vocab)] = new_tok

        # For every affected word, remove its pair contributions, apply merge, then re-add
        affected = list(pair_to_words[best])
        for wi in affected:
            toks = words[wi]
            freq = word_freqs[wi]
            # remove this word's pair contributions
            for i in range(len(toks) - 1):
                p = (toks[i], toks[i + 1])
                pair_counts[p] -= freq
                pair_to_words[p].discard(wi)
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                    pair_to_words.pop(p, None)

            # rebuild word applying merge
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

            # add new pair contributions
            for i in range(len(new_toks) - 1):
                p = (new_toks[i], new_toks[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(wi)

    return vocab, merges

from __future__ import annotations

import heapq
import os
import multiprocessing as mp
import regex as re
from collections import Counter, defaultdict
from typing import BinaryIO

from tqdm import tqdm

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(GPT2_PAT)

# Target byte size for sub-chunks fed to worker processes (~8 MB).
_PARALLEL_SUBCHUNK_BYTES = 8 * 1024 * 1024
_PARALLEL_MIN_TOTAL_BYTES = 4 * 1024 * 1024  # below this, stay serial


def _count_pretokens(chunk: str) -> Counter:
    counts: Counter = Counter()
    for match in PAT_RE.finditer(chunk):
        tok = match.group(0).encode("utf-8")
        counts[tuple(bytes([b]) for b in tok)] += 1
    return counts


def _split_on_newline_boundaries(chunk: str, target_bytes: int) -> list[str]:
    """Split a chunk into sub-chunks of ~target_bytes each, breaking only at \n.

    Splitting at \n is safe for the GPT-2 pretoken regex: \n is part of the
    \\s+ alternatives, but no GPT-2 pretoken match crosses a \n boundary in a
    way that would change tokenization when the chunk is split there.
    Specifically the regex never matches across \n: ' ?\\p{L}+', ' ?\\p{N}+',
    ' ?[^\\s\\p{L}\\p{N}]+' all stop at whitespace; '\\s+(?!\\S)' and '\\s+'
    are pure-whitespace runs but breaking them at \n still yields the same
    token sequence (each side becomes its own whitespace-run match).
    """
    n = len(chunk)
    if n <= target_bytes:
        return [chunk]
    # We work in characters here, not bytes. target_bytes is a soft target
    # treating 1 char ~ 1 byte; English-heavy text makes this close enough.
    parts: list[str] = []
    start = 0
    while start < n:
        end = start + target_bytes
        if end >= n:
            parts.append(chunk[start:])
            break
        nl = chunk.rfind("\n", start, end)
        if nl == -1 or nl <= start:
            # No newline in window — extend forward to the next newline.
            nl = chunk.find("\n", end)
            if nl == -1:
                parts.append(chunk[start:])
                break
        parts.append(chunk[start : nl + 1])
        start = nl + 1
    return parts


def _pretokenize_text_serial(text: str, special_tokens: list[str]) -> Counter:
    """Original single-threaded implementation. Kept for verification."""
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


def _pretokenize_text(
    text: str,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> Counter:
    """Split text on special tokens, then apply GPT-2 pretokenizer to each chunk.

    Parallelized across processes; deterministic and produces the same Counter
    as the serial implementation. Returns Counter of tuple[bytes].
    """
    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]

    # Build the list of sub-chunks to process. We split each special-bounded
    # chunk further on \n boundaries so workers see balanced workloads.
    sub_chunks: list[str] = []
    total_chars = 0
    for chunk in chunks:
        if not chunk:
            continue
        total_chars += len(chunk)
        sub_chunks.extend(_split_on_newline_boundaries(chunk, _PARALLEL_SUBCHUNK_BYTES))

    if not sub_chunks:
        return Counter()

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    # Stay serial for small inputs or when only one worker is requested.
    if num_workers <= 1 or total_chars < _PARALLEL_MIN_TOTAL_BYTES or len(sub_chunks) == 1:
        counts: Counter = Counter()
        for sc in sub_chunks:
            counts.update(_count_pretokens(sc))
        return counts

    ctx = mp.get_context("spawn")
    counts = Counter()
    with ctx.Pool(processes=min(num_workers, len(sub_chunks))) as pool:
        for partial in pool.imap_unordered(_count_pretokens, sub_chunks, chunksize=1):
            counts.update(partial)
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

    # Heap-backed priority queue with lazy deletion.
    #
    # Order is (count desc, pair-bytes desc) to match the original
    # `max(pair_counts, key=lambda p: (pair_counts[p], p))` tie-break exactly.
    # We push a fresh entry on every count increment; stale entries are
    # discarded on pop when their stored count no longer matches pair_counts.
    heap: list[tuple[int, _DescBytes, _DescBytes, tuple[bytes, bytes]]] = []
    for p, c in pair_counts.items():
        heapq.heappush(heap, (-c, _DescBytes(p[0]), _DescBytes(p[1]), p))

    target_merges = vocab_size - len(vocab)
    pbar = tqdm(total=target_merges, desc="bpe merges", unit="merge")
    for _ in range(target_merges):
        # Pop until we find an entry consistent with pair_counts.
        best: tuple[bytes, bytes] | None = None
        while heap:
            neg_count, _, _, candidate = heap[0]
            cur = pair_counts.get(candidate, 0)
            if cur <= 0 or -neg_count != cur:
                heapq.heappop(heap)
                continue
            best = candidate
            heapq.heappop(heap)
            break
        if best is None:
            break

        merges.append(best)
        new_tok = best[0] + best[1]
        vocab[len(vocab)] = new_tok

        # Snapshot affected word indices.
        affected = list(pair_to_words[best])
        b0, b1 = best
        # Track every pair whose count changed in this step so we can refresh
        # its heap entry once at the end (covers both increments and the
        # crucial decrement-without-deletion case).
        changed_pairs: set[tuple[bytes, bytes]] = set()
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
                changed_pairs.add(p)

            # rebuild word applying merge
            new_toks: list[bytes] = []
            i = 0
            n = len(toks)
            while i < n:
                if i < n - 1 and toks[i] == b0 and toks[i + 1] == b1:
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
                changed_pairs.add(p)

        # Re-push each changed pair with its final count. Stale entries with
        # higher counts will be discarded by the lazy check on pop.
        for p in changed_pairs:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, _DescBytes(p[0]), _DescBytes(p[1]), p))

        pbar.update(1)
    pbar.close()

    return vocab, merges


class _DescBytes:
    """Wrapper that inverts bytes ordering so heapq min-heap yields max-bytes first."""
    __slots__ = ("v",)

    def __init__(self, v: bytes) -> None:
        self.v = v

    def __lt__(self, other: "_DescBytes") -> bool:
        return self.v > other.v

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DescBytes) and self.v == other.v

    def __hash__(self) -> int:
        return hash(self.v)

from __future__ import annotations

import regex as re
from collections.abc import Iterable, Iterator
from typing import IO

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(GPT2_PAT)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # Ensure special tokens are in vocab
        existing = set(self.vocab.values())
        for s in self.special_tokens:
            b = s.encode("utf-8")
            if b not in existing:
                self.vocab[len(self.vocab)] = b
                existing.add(b)

        # inverse map
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        # merge rank: pair -> priority (lower = earlier)
        self._merge_rank: dict[tuple[bytes, bytes], int] = {p: i for i, p in enumerate(self.merges)}

        # Compile special token regex (longest first so "<|eot|><|eot|>" beats "<|eot|>")
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile("(" + "|".join(re.escape(s) for s in sorted_specials) + ")")
        else:
            self._special_re = None

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        import json

        with open(vocab_filepath) as f:
            raw = json.load(f)
        vocab = {int(k): v.encode("utf-8") for k, v in raw.items()}
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))
        return cls(vocab, merges, special_tokens)

    def _bpe_encode_word(self, token_bytes: bytes) -> list[int]:
        # Convert to list of single-byte tokens, then apply merges greedily by rank
        parts: list[bytes] = [bytes([b]) for b in token_bytes]
        if len(parts) < 2:
            return [self._bytes_to_id[p] for p in parts]

        while True:
            best_rank = None
            best_idx = -1
            for i in range(len(parts) - 1):
                rank = self._merge_rank.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            # merge at best_idx
            parts = parts[:best_idx] + [parts[best_idx] + parts[best_idx + 1]] + parts[best_idx + 2:]

        return [self._bytes_to_id[p] for p in parts]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self._special_re is not None:
            segments = self._special_re.split(text)
        else:
            segments = [text]

        special_set = set(self.special_tokens)
        for seg in segments:
            if not seg:
                continue
            if seg in special_set:
                ids.append(self._bytes_to_id[seg.encode("utf-8")])
                continue
            for match in PAT_RE.finditer(seg):
                tok_bytes = match.group(0).encode("utf-8")
                ids.extend(self._bpe_encode_word(tok_bytes))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buf = ""
        for chunk in iterable:
            buf += chunk
            # Find a safe split point — process up to the last newline if possible
            # to avoid splitting in the middle of a pre-token.
            last_newline = buf.rfind("\n")
            if last_newline >= 0:
                to_process = buf[: last_newline + 1]
                buf = buf[last_newline + 1:]
                for _id in self.encode(to_process):
                    yield _id
        if buf:
            for _id in self.encode(buf):
                yield _id

    def decode(self, ids: list[int]) -> str:
        parts = b"".join(self.vocab[i] for i in ids)
        return parts.decode("utf-8", errors="replace")

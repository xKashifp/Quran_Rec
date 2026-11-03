from __future__ import annotations

from pathlib import Path

import numpy as np

BLANK = "<blank>"


class Alphabet:
    def __init__(self, tokens_path: str | Path):
        rows = []
        for line in Path(tokens_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                sym, _, idx = line.rpartition(" ")
                rows.append((sym, int(idx)))
        self.tokens = {i: s for s, i in rows}
        self.blank_id = next(i for s, i in rows if s == BLANK)
        self.chars = sorted({c for s, _ in rows if s != BLANK for c in s})
        self.char_id = {c: i for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.char_id[c] for c in text if c in self.char_id], dtype=np.int64)

    def encode_tokens(self, tokens: list[str]) -> np.ndarray:
        return self.encode("".join(t for t in tokens if t != BLANK))

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

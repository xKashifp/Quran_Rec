from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phonemes import Alphabet


@dataclass
class Ayah:
    surah: int
    ayah: int
    text: str
    phoneme_text: str
    phonemes: np.ndarray


def load_quran(text_path: Path, reference_path: Path, alphabet: Alphabet):
    text = json.loads(text_path.read_text(encoding="utf-8"))
    surahs = {s["number"]: s for s in text["surahs"]}
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    ayahs = []
    for a in text["ayahs"]:
        key = f"{a['surah']}:{a['ayah']}"
        if key not in reference:
            raise ValueError(f"{key} missing from {reference_path.name}")
        ph = reference[key]["aya_phoneme"]
        ayahs.append(Ayah(a["surah"], a["ayah"], a["text"], ph, alphabet.encode(ph)))

    unknown = sorted({c for a in ayahs for c in a.phoneme_text if c not in alphabet.char_id and not c.isspace()})
    if unknown:
        raise ValueError(f"reference uses characters not in tokens.txt: {unknown}")
    return ayahs, surahs

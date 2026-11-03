import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from matcher import QuranMatcher
from phonemes import Alphabet
from quran import load_quran
from recognizer import SAMPLE_RATE, Recognizer, load_audio

MODEL = ROOT / "models" / "zipformer_p-arabic-v3"
AUDIO = ROOT / "tests" / "audio"
RECITERS = ["Alafasy_128kbps", "Husary_128kbps", "Abdul_Basit_Murattal_192kbps", "Minshawy_Murattal_128kbps"]

FULL = [(1, 2), (2, 255), (18, 10), (36, 1), (55, 13), (67, 1), (112, 1), (114, 4), (3, 26), (24, 35)]
PARTIAL = [(2, 255), (24, 35), (2, 282), (18, 10), (3, 190)]
CROSS = [((1, 6), (1, 7)), ((112, 3), (112, 4)), ((67, 1), (67, 2)), ((36, 2), (36, 3))]


def fetch(reciter, s, a):
    path = AUDIO / reciter / f"{s:03d}{a:03d}.mp3"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://everyayah.com/data/{reciter}/{s:03d}{a:03d}.mp3", path)
    return load_audio(path.read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zipformer_p_arabic_v3.1.onnx")
    ap.add_argument("--reciters", type=int, default=2)
    args = ap.parse_args()

    alphabet = Alphabet(MODEL / "tokens.txt")
    ayahs, _ = load_quran(ROOT / "data" / "quran_text.json", MODEL / "ordered_quran_phonemes.json", alphabet)
    key = {(a.surah, a.ayah): i for i, a in enumerate(ayahs)}
    matcher = QuranMatcher([a.phonemes for a in ayahs])
    rec = Recognizer(MODEL / args.model, MODEL / "tokens.txt")

    cases = []
    for r in RECITERS[: args.reciters]:
        for s, a in FULL:
            cases.append(("full", r, f"{s}:{a}", {key[(s, a)]}, lambda r=r, s=s, a=a: fetch(r, s, a)))
        for s, a in PARTIAL:
            def part(r=r, s=s, a=a):
                x = fetch(r, s, a)
                return x[int(len(x) * 0.3): int(len(x) * 0.65)]
            cases.append(("partial", r, f"{s}:{a} 30-65%", {key[(s, a)]}, part))
        for (s1, a1), (s2, a2) in CROSS:
            def cross(r=r, s1=s1, a1=a1, s2=s2, a2=a2):
                x, y = fetch(r, s1, a1), fetch(r, s2, a2)
                return np.concatenate([x[len(x) // 2:], y[: len(y) // 2]])
            cases.append(("cross", r, f"{s1}:{a1}+{s2}:{a2}", {key[(s1, a1)], key[(s2, a2)]}, cross))

    stats = {}
    for kind, r, label, truth, get in cases:
        try:
            x = get()
        except Exception as exc:
            print(f"skip {r} {label}: {exc}")
            continue
        t0 = time.time()
        ids = alphabet.encode_tokens(rec.transcribe(x))
        t1 = time.time()
        cands = matcher.search(ids)
        t2 = time.time()
        hits = [bool(truth & set(c.ayahs)) for c in cands]
        ok1, ok3 = bool(hits and hits[0]), any(hits)
        st = stats.setdefault(kind, [0, 0, 0])
        st[0] += 1; st[1] += ok1; st[2] += ok3
        top = cands[0] if cands else None
        got = f"{ayahs[top.primary].surah}:{ayahs[top.primary].ayah} {top.score:.2f}" if top else "-"
        print(f"{'OK ' if ok1 else ('top3' if ok3 else 'MISS')} {kind:7s} {r[:14]:14s} {label:16s} -> {got:12s} "
              f"{len(x) / SAMPLE_RATE:5.1f}s audio, asr {1000 * (t1 - t0):5.0f} ms, search {1000 * (t2 - t1):4.0f} ms")
    print()
    for kind, (n, a1, a3) in stats.items():
        print(f"{kind:8s} top1 {a1}/{n}  top3 {a3}/{n}")


if __name__ == "__main__":
    main()

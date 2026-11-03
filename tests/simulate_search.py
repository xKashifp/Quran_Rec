import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from matcher import QuranMatcher


def stand_in_corpus():
    data = json.loads((ROOT / "data" / "quran_text.json").read_text(encoding="utf-8"))
    chars = {}
    seqs = []
    for a in data["ayahs"]:
        ids = [chars.setdefault(c, len(chars)) for c in a["text"].replace(" ", "")]
        seqs.append(np.array(ids, dtype=np.int64))
    return seqs, len(chars)


def corrupt(seq, rng, err, vocab):
    out = []
    for t in seq:
        r = rng.random()
        if r < err / 3:
            continue
        if r < 2 * err / 3:
            out.append(int(rng.integers(vocab)))
        else:
            out.append(int(t))
        if rng.random() < err / 3:
            out.append(int(rng.integers(vocab)))
        if rng.random() < 0.05:
            out.append(int(t))
    return np.array(out, dtype=np.int64)


def make_query(seqs, starts, rng, kind):
    n = len(seqs)
    if kind == "full":
        a = int(rng.integers(n))
        return seqs[a], {a}
    if kind == "partial":
        a = int(rng.integers(n))
        while len(seqs[a]) < 40:
            a = int(rng.integers(n))
        L = len(seqs[a])
        ln = int(rng.integers(max(15, L // 4), max(16, L // 2 + 1)))
        s = int(rng.integers(0, L - ln + 1))
        return seqs[a][s:s + ln], {a}
    a = int(rng.integers(n - 1))
    tail = seqs[a][len(seqs[a]) // 2:]
    head = seqs[a + 1][: max(10, len(seqs[a + 1]) // 2)]
    return np.concatenate([tail, head]), {a, a + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--err", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seqs, vocab = stand_in_corpus()
    t0 = time.time()
    m = QuranMatcher(seqs)
    print(f"index built in {time.time() - t0:.2f}s, corpus {len(m.corpus)} tokens, vocab {vocab}")
    rng = np.random.default_rng(args.seed)
    for kind in ("full", "partial", "cross"):
        top1 = top3 = 0
        times = []
        for _ in range(args.n):
            clean, truth = make_query(seqs, None, rng, kind)
            q = corrupt(clean, rng, args.err, vocab)
            t = time.time()
            cands = m.search(q)
            times.append(time.time() - t)
            hit = [bool(truth & set(c.ayahs)) for c in cands]
            top1 += bool(hit[:1] and hit[0])
            top3 += any(hit)
        print(f"{kind:8s} top1 {top1 / args.n:6.1%}  top3 {top3 / args.n:6.1%}  "
              f"median {np.median(times) * 1000:5.0f} ms  max {max(times) * 1000:5.0f} ms")


if __name__ == "__main__":
    main()

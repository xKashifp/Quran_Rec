from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

REPEAT_COST = 0.3
GAP_COST = 1.0
SUB_COST = 1.0


def collapse_runs(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(seq) == 0:
        return seq.copy(), np.zeros(0, dtype=np.int64)
    keep = np.ones(len(seq), dtype=bool)
    keep[1:] = seq[1:] != seq[:-1]
    starts = np.flatnonzero(keep)
    return seq[starts], starts


def _kgram_hashes(seq: np.ndarray, k: int) -> np.ndarray:
    n = len(seq) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    s = seq.astype(np.int64)
    h = np.zeros(n, dtype=np.int64)
    for t in range(k):
        h = (h << 8) | s[t:t + n]
    return h


def fit_align(q: np.ndarray, r: np.ndarray, sub_matrix: np.ndarray | None = None,
              anchored_start: bool = False) -> tuple[float, int]:
    m, n = len(q), len(r)
    if m == 0:
        return 0.0, 0
    rskip = np.full(n, GAP_COST)
    if n > 1:
        rskip[1:][r[1:] == r[:-1]] = REPEAT_COST
    C = np.concatenate(([0.0], np.cumsum(rskip)))
    prev = C.copy() if anchored_start else np.zeros(n + 1)
    A = np.empty(n + 1)
    for i in range(m):
        qdel = REPEAT_COST if i > 0 and q[i] == q[i - 1] else GAP_COST
        if sub_matrix is not None:
            sub = sub_matrix[q[i], r]
        else:
            sub = (r != q[i]) * SUB_COST
        A[0] = prev[0] + qdel
        np.minimum(prev[:-1] + sub, prev[1:] + qdel, out=A[1:])
        prev = C + np.minimum.accumulate(A - C)
    end = int(np.argmin(prev))
    return float(prev[end]), end


def align_span(q: np.ndarray, r: np.ndarray, sub_matrix=None) -> tuple[float, int, int]:
    cost, end = fit_align(q, r, sub_matrix)
    _, length = fit_align(q[::-1], r[:end][::-1], sub_matrix, anchored_start=True)
    return cost, end - length, end


@dataclass
class Candidate:
    score: float
    start: int
    end: int
    primary: int
    ayahs: list[int]
    coverage: float
    votes: float


class QuranMatcher:
    def __init__(self, ayah_tokens: list[np.ndarray], sub_matrix: np.ndarray | None = None,
                 ks: tuple[int, ...] = (3, 4)):
        self.sub_matrix = sub_matrix
        self.ayah_len = np.array([len(a) for a in ayah_tokens], dtype=np.int64)
        self.ayah_start = np.concatenate(([0], np.cumsum(self.ayah_len)))
        self.corpus = np.concatenate(ayah_tokens).astype(np.int16)
        self.ayah_of = np.repeat(np.arange(len(ayah_tokens)), self.ayah_len)
        self.cc, self.c2o = collapse_runs(self.corpus)
        self.index = {}
        for k in ks:
            h = _kgram_hashes(self.cc, k)
            order = np.argsort(h, kind="stable")
            self.index[k] = (h[order], order)

    def _peaks(self, qc: np.ndarray, n_peaks: int) -> list[tuple[int, float]]:
        mq = len(qc)
        k = 4 if mq >= 12 else 3
        sorted_h, order = self.index[k]
        qh = _kgram_hashes(qc, k)
        lo = np.searchsorted(sorted_h, qh, "left")
        hi = np.searchsorted(sorted_h, qh, "right")
        cnt = hi - lo
        ok = (cnt > 0) & (cnt <= 4000)
        qpos = np.arange(len(qh))[ok]
        lo, cnt = lo[ok], cnt[ok]
        if cnt.sum() == 0:
            return []
        w = 1.0 / (1.0 + np.log(cnt))
        total = int(cnt.sum())
        offs = np.repeat(lo - np.cumsum(cnt) + cnt, cnt) + np.arange(total)
        ref_pos = order[offs]
        diag = ref_pos - np.repeat(qpos, cnt) + mq
        votes = np.bincount(diag, weights=np.repeat(w, cnt), minlength=len(self.cc) + mq + 1)
        band = max(2, int(0.08 * mq))
        cs = np.concatenate(([0.0], np.cumsum(votes)))
        idx = np.arange(len(votes))
        smooth = cs[np.minimum(idx + band + 1, len(votes))] - cs[np.maximum(idx - band, 0)]
        top = np.argpartition(-smooth, min(len(smooth) - 1, 4000))[:4000]
        top = top[np.argsort(-smooth[top])]
        sep = max(8, mq // 2)
        chosen: list[tuple[int, float]] = []
        for d in top:
            if smooth[d] <= 0:
                break
            if all(abs(int(d) - c) > sep for c, _ in chosen):
                chosen.append((int(d), float(smooth[d])))
                if len(chosen) >= n_peaks:
                    break
        return [(d - mq, v) for d, v in chosen]

    def _span_to_ayahs(self, s: int, e: int) -> tuple[int, list[int], float]:
        if e <= s:
            a = int(self.ayah_of[min(s, len(self.corpus) - 1)])
            return a, [a], 0.0
        ids, counts = np.unique(self.ayah_of[s:e], return_counts=True)
        lens = self.ayah_len[ids]
        keep = (counts >= np.maximum(5, 0.25 * lens)) | (counts >= 0.6 * lens)
        if not keep.any():
            keep = counts == counts.max()
        primary = int(ids[np.argmax(counts)])
        cover = float(counts.max() / self.ayah_len[primary])
        return primary, [int(a) for a in ids[keep]], cover

    def search(self, query: np.ndarray | list[int], top_k: int = 3) -> list[Candidate]:
        q = np.asarray(query, dtype=np.int64)
        qc, _ = collapse_runs(q)
        if len(qc) < 3:
            return []
        n_peaks = 24 if len(q) < 250 else 12
        peaks = self._peaks(qc, n_peaks)
        mq = len(qc)
        margin = max(12, int(0.3 * mq))
        best: dict[int, Candidate] = {}
        n_c = len(self.cc)
        for d, v in peaks:
            cs, ce = max(0, d - margin), min(n_c, d + mq + margin)
            if cs >= ce:
                continue
            os_ = int(self.c2o[cs])
            oe = int(self.c2o[ce]) if ce < n_c else len(self.corpus)
            cost, s, e = align_span(q, self.corpus[os_:oe].astype(np.int64), self.sub_matrix)
            score = max(0.0, 1.0 - cost / len(q))
            primary, ayahs, cover = self._span_to_ayahs(os_ + s, os_ + e)
            cand = Candidate(score, os_ + s, os_ + e, primary, ayahs, cover, v)
            if primary not in best or score > best[primary].score:
                best[primary] = cand
        ranked = sorted(best.values(), key=lambda c: (-c.score, c.start))
        return ranked[:top_k]


def confidence_label(cands: list[Candidate]) -> str:
    if not cands:
        return "none"
    top = cands[0].score
    gap = top - cands[1].score if len(cands) > 1 else top
    if top >= 0.8 and gap >= 0.1:
        return "high"
    if top >= 0.6 and gap >= 0.05:
        return "medium"
    if math.isclose(gap, 0.0, abs_tol=1e-6) and top >= 0.8:
        return "ambiguous (identical wording occurs more than once)"
    return "low"

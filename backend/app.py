from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from matcher import QuranMatcher, confidence_label
from phonemes import Alphabet
from quran import load_quran
from recognizer import SAMPLE_RATE, Recognizer, load_audio

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("QURANLAB_MODEL_DIR", ROOT / "models" / "zipformer_p-arabic-v3"))
MODEL_FILE = os.environ.get("QURANLAB_MODEL_FILE", "zipformer_p_arabic_v3.1.onnx")
TEXT_PATH = ROOT / "data" / "quran_text.json"
MAX_SECONDS = 90
RECORDINGS = ROOT / "recordings"
SAVE_RECORDINGS = os.environ.get("QURANLAB_SAVE_RECORDINGS", "0") == "1"

state: dict = {"ready": False, "error": None}


def startup() -> None:
    try:
        model = MODEL_DIR / MODEL_FILE
        tokens = MODEL_DIR / "tokens.txt"
        reference = MODEL_DIR / "ordered_quran_phonemes.json"
        for p in (model, tokens, reference, TEXT_PATH):
            if not p.exists():
                raise FileNotFoundError(f"missing {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}. "
                                        "Run scripts/download_model.py and scripts/build_quran_text.py first.")
        t0 = time.time()
        alphabet = Alphabet(tokens)
        ayahs, surahs = load_quran(TEXT_PATH, reference, alphabet)
        matcher = QuranMatcher([a.phonemes for a in ayahs])
        recognizer = Recognizer(model, tokens, num_threads=os.cpu_count() or 4)
        state.update(alphabet=alphabet, ayahs=ayahs, surahs=surahs, matcher=matcher,
                     recognizer=recognizer, ready=True,
                     load_seconds=round(time.time() - t0, 2))
        print(f"ready: {MODEL_FILE}, {len(ayahs)} ayat, {len(matcher.corpus)} phonemes, "
              f"{len(alphabet)} symbols, loaded in {state['load_seconds']}s")
    except Exception as exc:
        state["error"] = str(exc)
        print(f"NOT READY: {exc}")


@asynccontextmanager
async def lifespan(_app):
    startup()
    yield


app = FastAPI(title="Quran Ayah Finder", lifespan=lifespan)


def ayah_info(i: int) -> dict:
    a = state["ayahs"][i]
    s = state["surahs"][a.surah]
    return {"surah": a.surah, "ayah": a.ayah, "surah_name_ar": s["name_ar"],
            "surah_name_en": s["name_en"], "surah_translation": s["translation"], "text": a.text}


@app.get("/api/health")
def health():
    if not state["ready"]:
        return {"ready": False, "error": state["error"]}
    return {"ready": True, "model": MODEL_FILE, "ayahs": len(state["ayahs"]),
            "phonemes": int(len(state["matcher"].corpus)), "symbols": len(state["alphabet"]),
            "load_seconds": state["load_seconds"]}


@app.post("/api/recognize")
async def recognize(audio: UploadFile = File(...)):
    if not state["ready"]:
        raise HTTPException(503, f"Model not loaded: {state['error']}")
    data = await audio.read()
    try:
        samples = load_audio(data)
    except Exception as exc:
        raise HTTPException(400, f"Could not read audio: {exc}")
    seconds = len(samples) / SAMPLE_RATE
    if seconds < 0.5:
        raise HTTPException(400, "Recording is too short.")
    if seconds > MAX_SECONDS:
        samples = samples[: MAX_SECONDS * SAMPLE_RATE]

    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak < 1e-3:
        raise HTTPException(400, "The recording is silent. Check that your microphone is on and selected in the browser.")
    if 0 < peak < 0.3:
        samples = samples * (0.3 / peak)

    t0 = time.time()
    tokens = state["recognizer"].transcribe(samples)
    t1 = time.time()
    alphabet = state["alphabet"]
    ids = alphabet.encode_tokens(tokens)
    cands = state["matcher"].search(ids, top_k=3)
    t2 = time.time()

    top = cands[0] if cands else None
    summary = (f"{state['ayahs'][top.primary].surah}:{state['ayahs'][top.primary].ayah} score {top.score:.2f}"
               if top else "no match")
    print(f"recognize: {seconds:.1f}s audio, peak {peak:.3f}, {len(ids)} phonemes -> {summary}")
    if SAVE_RECORDINGS:
        RECORDINGS.mkdir(exist_ok=True)
        stem = RECORDINGS / time.strftime("%Y%m%d-%H%M%S")
        sf.write(stem.with_suffix(".wav"), samples, SAMPLE_RATE)
        stem.with_suffix(".txt").write_text(summary + "\n" + alphabet.decode(ids) + "\n", encoding="utf-8")

    results = []
    for c in cands:
        covered = [ayah_info(i) for i in c.ayahs]
        results.append({
            "score": round(c.score, 4),
            "primary": ayah_info(c.primary),
            "ayahs": covered,
            "coverage": round(min(c.coverage, 1.0), 3),
            "reference_phonemes": alphabet.decode(state["matcher"].corpus[c.start:c.end]),
        })
    return {
        "duration_seconds": round(seconds, 2),
        "recognized_phonemes": alphabet.decode(ids),
        "phoneme_count": len(ids),
        "confidence": confidence_label(cands),
        "results": results,
        "timing_ms": {"recognition": round((t1 - t0) * 1000), "search": round((t2 - t1) * 1000)},
    }


FRONTEND = ROOT / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")

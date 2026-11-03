from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def load_audio(data: bytes) -> np.ndarray:
    samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if sr != SAMPLE_RATE:
        samples = resample(samples, sr, SAMPLE_RATE)
    return np.ascontiguousarray(samples, dtype=np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if len(x) == 0:
        return x
    if sr_in > sr_out:
        width = max(1, int(round(sr_in / sr_out)))
        kernel = np.hanning(4 * width + 1)
        x = np.convolve(x, kernel / kernel.sum(), mode="same")
    n_out = int(round(len(x) * sr_out / sr_in))
    t_out = np.arange(n_out) * (sr_in / sr_out)
    return np.interp(t_out, np.arange(len(x)), x).astype(np.float32)


class Recognizer:
    def __init__(self, model: Path, tokens: Path, num_threads: int = 4):
        import sherpa_onnx

        self.model_name = model.name
        self._rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=str(tokens),
            model=str(model),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        self._lock = threading.Lock()

    def transcribe(self, samples: np.ndarray) -> list[str]:
        with self._lock:
            stream = self._rec.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            stream.accept_waveform(SAMPLE_RATE, np.zeros(int(0.8 * SAMPLE_RATE), dtype=np.float32))
            stream.input_finished()
            while self._rec.is_ready(stream):
                self._rec.decode_stream(stream)
            return list(self._rec.tokens(stream))

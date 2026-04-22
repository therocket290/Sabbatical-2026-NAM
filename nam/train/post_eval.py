from __future__ import annotations
import math
import csv
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from nam.data import wav_to_tensor, np_to_wav
from nam.models.losses import esr as nam_esr
import soundfile as sf


def _to_numpy_mono(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).squeeze()

def _remove_dc(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)

def _safe_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def _esr_np(ref: np.ndarray, est: np.ndarray) -> float:
    ref_t = torch.tensor(ref, dtype=torch.float32)
    est_t = torch.tensor(est, dtype=torch.float32)
    return float(nam_esr(est_t, ref_t).item())

def _align_by_xcorr(ref: np.ndarray, est: np.ndarray, max_shift: int = 4000):
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    best_shift = 0
    best_score = -np.inf
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            r, e = ref[: n - shift], est[shift:n]
        else:
            r, e = ref[-shift:n], est[: n + shift]
        score = float(np.dot(r, e))
        if score > best_score:
            best_score, best_shift = score, shift
    if best_shift >= 0:
        return ref[: n - best_shift], est[best_shift:n], best_shift
    return ref[-best_shift:n], est[: n + best_shift], best_shift


def _calculate_asr_metrics_sato(
    signal: np.ndarray,
    f0: float,
    sr: int = 48017,
    N: int = 48017,
    discard_seconds: float = 0.5,
) -> Dict[str, float]:
    x = np.asarray(signal, dtype=np.float64).squeeze()

    start = int(round(discard_seconds * sr))
    if len(x) < start + N:
        raise ValueError(
            f"Signal too short for ASR calculation: need at least {start + N} "
            f"samples, got {len(x)}."
        )

    x = x[start:start + N]

    # no window, no zero-padding
    Y = np.fft.rfft(x)
    power = np.abs(Y) ** 2

    k0_float = f0 * N / sr
    k0 = int(round(k0_float))

    if not np.isclose(k0_float, k0, atol=1e-10):
        raise ValueError(
            f"f0={f0} is not an exact DFT bin for sr={sr}, N={N}. "
            f"k0={k0_float:.12f}"
        )

    if math.gcd(k0, N) != 1:
        raise ValueError(f"k0={k0} and N={N} are not coprime.")

    N0 = (N - 1) // (2 * k0)
    harmonic_bins = np.arange(1, N0 + 1, dtype=int) * k0

    EH = float(np.sum(power[harmonic_bins]))
    EY = float(np.sum(power))
    EA = max(EY - EH, 0.0)

    if EH <= 0.0:
        return {
            "ASR_sato": 1.0,
            "SADR_sato": 0.0,
            "EH_sato": EH,
            "EA_sato": EA,
            "EY_sato": EY,
            "k0_sato": k0,
            "Nharm_sato": int(N0),
        }

    return {
        "ASR_sato": EA / EH,
        "SADR_sato": 10.0 * np.log10(EH / (EA + 1e-12)),
        "EH_sato": EH,
        "EA_sato": EA,
        "EY_sato": EY,
        "k0_sato": k0,
        "Nharm_sato": int(N0),
    }


def evaluate_case_standard(ref_path, est_path, sr=48000):
    if not ref_path.exists() or not est_path.exists():
        return {}

    ref = _remove_dc(_to_numpy_mono(wav_to_tensor(ref_path, rate=sr)))
    est = _remove_dc(_to_numpy_mono(wav_to_tensor(est_path, rate=sr)))
    ref_al, est_al, _ = _align_by_xcorr(ref, est)

    return {
        "ESR": _esr_np(ref_al, est_al),
        "ref_rms": _safe_rms(ref_al),
        "est_rms": _safe_rms(est_al),
    }


def evaluate_case_sine_sato(est_path, f0, sr=48017):
    if not est_path.exists():
        return {}

    est = _remove_dc(_to_numpy_mono(wav_to_tensor(est_path, rate=sr)))

    out = {
        "est_rms": _safe_rms(est),
    }
    out.update(_calculate_asr_metrics_sato(est, f0=f0, sr=sr, N=48017, discard_seconds=0.5))
    return out

 # def _render_cases(model, test_dir: Path, out_dir: Path, cases: Dict[str, str]):
 #   out_dir.mkdir(parents=True, exist_ok=True)
 #   for name, di_name in cases.items():
 #       x = wav_to_tensor(test_dir / di_name)
  #      with torch.no_grad():
 #           y_pred = model(x).flatten().cpu().numpy()
 #       np_to_wav(y_pred, out_dir / f"{name}_pred.wav")

def _write_wav(path: Path, y: np.ndarray, sr: int):
    y = np.asarray(y, dtype=np.float32)
    peak = np.max(np.abs(y))
    if peak > 0.999:
        y = (y / peak * 0.999).astype(np.float32)
    sf.write(path, y, sr, subtype="PCM_24")


def _render_cases(model, test_dir: Path, out_dir: Path, cases: Dict[str, str], sr: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, di_name in cases.items():
        x = wav_to_tensor(test_dir / di_name, rate=sr)
        with torch.no_grad():
            y_pred = model(x).flatten().cpu().numpy()
        _write_wav(out_dir / f"{name}_pred.wav", y_pred, sr)

def run_test_set(
    model,
    standard_test_dir="tests_48k",
    sine_test_dir="tests_48017",
    out_dir_standard="test_predictions_48k",
    out_dir_sine="test_predictions_48017",
    csv_path="experiment_results_asr_sato.csv",
    run_metadata: Optional[Dict] = None,
):
    standard_test_dir = Path(standard_test_dir)
    sine_test_dir = Path(sine_test_dir)
    out_dir_standard = Path(out_dir_standard)
    out_dir_sine = Path(out_dir_sine)
    csv_path = Path(csv_path)

    standard_cases = {
        "sweep": "sweep_DI.wav",
        "playing": "playing_DI.wav",
    }

    sine_cases = {
        "sine_soft_mid": "sine_soft_mid_DI.wav",
        "sine_loud_mid": "sine_loud_mid_DI.wav",
        "sine_soft_high": "sine_soft_high_DI.wav",
        "sine_loud_high": "sine_loud_high_DI.wav",
    }

    sine_freqs = {
        "sine_soft_mid": 1249.0,
        "sine_loud_mid": 1249.0,
        "sine_soft_high": 5003.0,
        "sine_loud_high": 5003.0,
    }


    _render_cases(model, standard_test_dir, out_dir_standard, standard_cases, sr=48000)
    _render_cases(model, sine_test_dir, out_dir_sine, sine_cases, sr=48017)

    results = {}

    for name in standard_cases:
        results[name] = evaluate_case_standard(
            standard_test_dir / f"{name}_ref.wav",
            out_dir_standard / f"{name}_pred.wav",
            sr=48000,
        )

    for name in sine_cases:
        results[name] = evaluate_case_sine_sato(
            out_dir_sine / f"{name}_pred.wav",
            f0=sine_freqs[name],
            sr=48017,
        )

    row = {}
    if run_metadata:
        row.update(run_metadata)

    for case_name, metrics in results.items():
        for metric_name, value in metrics.items():
            row[f"{case_name}_{metric_name}"] = value

    fieldnames = list(row.keys())
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return results

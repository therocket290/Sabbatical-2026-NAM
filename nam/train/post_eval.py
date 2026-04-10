from __future__ import annotations
import torch
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from nam.data import wav_to_tensor, np_to_wav
from nam.models.losses import esr as nam_esr

def _to_numpy_mono(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64).squeeze()
    return x

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
    else:
        return ref[-best_shift:n], est[: n + best_shift], best_shift

def _calculate_asr_metrics(
    signal: np.ndarray, 
    f_ins: List[float], 
    sr: int = 48000, 
    num_harmonics: int = 20, 
    tolerance_hz: float = 2.0  # Tight tolerance for prime N
) -> Dict[str, float]:
    """
    Implements ASR from Sato & Smith (2025). 
    Uses prime N=48017 for maximum bin separation.
    """
    from scipy.signal.windows import hann
    from numpy.fft import rfft, rfftfreq
    
    # Use prime length N from paper (approx 1 second)
    # Slicing from the middle of your 2s clip to avoid transients
    N = 48017 
    if len(signal) > N + sr:
        start = sr # Start at 1.0 seconds
        signal = signal[start : start + N]
    else:
        signal = signal[:N]

    window = hann(len(signal))
    spec = np.abs(rfft(signal * window))
    freqs = rfftfreq(len(signal), 1/sr)
    
    is_harmonic = np.zeros_like(freqs, dtype=bool)
    for f_in in f_ins:
        for h in range(1, num_harmonics + 1):
            target_f = f_in * h
            if target_f > sr / 2: break
            is_harmonic |= (np.abs(freqs - target_f) < tolerance_hz)
            
    is_aliasing = (~is_harmonic) & (freqs > 20)
    
    sig_energy = np.sum(spec[is_harmonic]**2)
    alias_energy = np.sum(spec[is_aliasing]**2)
    
    if sig_energy == 0: return {"SADR": 0.0, "ASR": 1.0}
    
    # ASR is the linear ratio used in the paper (Aliasing / Signal)
    return {
        "SADR": 10 * np.log10(sig_energy / (alias_energy + 1e-12)),
        "ASR": alias_energy / sig_energy
    }

def evaluate_case(ref_path, est_path, sr=48000, sine_frequencies=None):
    ref = _remove_dc(_to_numpy_mono(wav_to_tensor(ref_path, rate=sr)))
    est = _remove_dc(_to_numpy_mono(wav_to_tensor(est_path, rate=sr)))
    ref_al, est_al, shift = _align_by_xcorr(ref, est)
    
    out = {
        "ESR": _esr_np(ref_al, est_al),
        "ref_rms": _safe_rms(ref_al),
        "est_rms": _safe_rms(est_al),
    }

    if sine_frequencies:
        metrics = _calculate_asr_metrics(est_al, sine_frequencies, sr=sr)
        out.update(metrics)
    return out

def run_test_set(model, test_dir="tests", out_dir="test_predictions", csv_path="experiment_results.csv"):
    test_dir, out_dir = Path(test_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # 1. Update the rendering list to use your new 2x2 matrix files
    tests = [
        ("sine_soft_mid", "sine_soft_mid_DI.wav"),
        ("sine_loud_mid", "sine_loud_mid_DI.wav"),
        ("sine_soft_high", "sine_soft_high_DI.wav"),
        ("sine_loud_high", "sine_loud_high_DI.wav"),
        ("sweep", "sweep_DI.wav"),
        ("playing", "playing_DI.wav"),
    ]

    for name, di in tests:
        x = wav_to_tensor(test_dir / di)
        with torch.no_grad():
            y_pred = model(x).flatten().cpu().numpy()
        np_to_wav(y_pred, out_dir / f"{name}_pred.wav")

    # 2. Define the evaluation cases with your Prime Frequencies
    cases = {
        "sine_soft_mid":  {"freqs": [1249]},
        "sine_loud_mid":  {"freqs": [1249]},
        "sine_soft_high": {"freqs": [5003, 7001]},
        "sine_loud_high": {"freqs": [5003, 7001]},
        "sweep": {"freqs": None},
        "playing": {"freqs": None},
    }

    row = {}
    for name, cfg in cases.items():
        m = evaluate_case(test_dir / f"{name}_ref.wav", out_dir / f"{name}_pred.wav", sine_frequencies=cfg["freqs"])
        for k, v in m.items(): row[f"{name}_{k}"] = v

    # Write to CSV
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header: writer.writeheader()
        writer.writerow(row)
    return row
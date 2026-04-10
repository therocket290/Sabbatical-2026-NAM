from __future__ import annotations
import torch
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from nam.data import wav_to_tensor, np_to_wav
from nam.models.losses import esr as nam_esr

# --- Core Math Helpers ---

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
    tolerance_hz: float = 2.0 
) -> Dict[str, float]:
    from scipy.signal.windows import hann
    from numpy.fft import rfft, rfftfreq
    
    # Prime N from Sato & Smith paper for bin separation
    N = 48017 
    # Grab the window starting at 0.5s to allow model settling
    start = int(sr * 0.5)
    if len(signal) >= (start + N):
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
    return {
        "SADR": 10 * np.log10(sig_energy / (alias_energy + 1e-12)),
        "ASR": alias_energy / sig_energy
    }

# --- Evaluation Logic ---

def evaluate_case(ref_path, est_path, sr=48000, sine_frequencies=None):
    if not ref_path.exists(): return {}
    ref = _remove_dc(_to_numpy_mono(wav_to_tensor(ref_path, rate=sr)))
    est = _remove_dc(_to_numpy_mono(wav_to_tensor(est_path, rate=sr)))
    ref_al, est_al, shift = _align_by_xcorr(ref, est)
    
    out = {
        "ESR": _esr_np(ref_al, est_al),
        "ref_rms": _safe_rms(ref_al),
        "est_rms": _safe_rms(est_al),
    }
    if sine_frequencies:
        out.update(_calculate_asr_metrics(est_al, sine_frequencies, sr=sr))
    return out

def evaluate_predictions(
    test_dir: str | Path = "tests",
    pred_dir: str | Path = "test_predictions",
    csv_path: str | Path = "experiment_results.csv",
    sr: int = 48_000,
    run_metadata: Optional[Dict] = None,
):
    test_dir, pred_dir, csv_path = Path(test_dir), Path(pred_dir), Path(csv_path)

    cases = {
        "sine_soft_mid":  {"ref": "sine_soft_mid_ref.wav",  "freqs": [1249]},
        "sine_loud_mid":  {"ref": "sine_loud_mid_ref.wav",  "freqs": [1249]},
        "sine_soft_high": {"ref": "sine_soft_high_ref.wav", "freqs": [5003]},
        "sine_loud_high": {"ref": "sine_loud_high_ref.wav", "freqs": [5003]},
        "sweep":          {"ref": "sweep_ref.wav",          "freqs": None},
        "playing":        {"ref": "playing_ref.wav",        "freqs": None},
    }

    results = {}
    for name, cfg in cases.items():
        results[name] = evaluate_case(
            test_dir / cfg["ref"], 
            pred_dir / f"{name}_pred.wav", 
            sr=sr, 
            sine_frequencies=cfg["freqs"]
        )
        
    # Flatten into a single row, starting with metadata
    row = {}
    if run_metadata:
        row.update(run_metadata)

    for case_name, metrics in results.items():
        for metric_name, value in metrics.items():
            row[f"{case_name}_{metric_name}"] = value

    # CSV Writing Logic (matches your provided file)
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return results

# --- Rendering & Orchestration ---

def run_test_set(model, test_dir="tests", out_dir="test_predictions", csv_path="experiment_results.csv", run_metadata=None):
    test_dir, out_dir = Path(test_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Rendering the 2x2 matrix + standard tests
    tests = [
        ("sine_soft_mid",  "sine_soft_mid_DI.wav"),
        ("sine_loud_mid",  "sine_loud_mid_DI.wav"),
        ("sine_soft_high", "sine_soft_high_DI.wav"),
        ("sine_loud_high", "sine_loud_high_DI.wav"),
        ("sweep",          "sweep_DI.wav"),
        ("playing",        "playing_DI.wav"),
    ]

    for name, di in tests:
        x = wav_to_tensor(test_dir / di)
        with torch.no_grad():
            y_pred = model(x).flatten().cpu().numpy()
        np_to_wav(y_pred, out_dir / f"{name}_pred.wav")

    return evaluate_predictions(
        test_dir=test_dir, 
        pred_dir=out_dir, 
        csv_path=csv_path, 
        run_metadata=run_metadata
    )
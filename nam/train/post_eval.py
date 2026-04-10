from __future__ import annotations
##########################
import torch
from pathlib import Path
from nam.data import wav_to_tensor, np_to_wav
        
###############

from pathlib import Path
from typing import Dict, Optional
import csv

import numpy as np
import torch

from nam.data import wav_to_tensor, np_to_wav
from nam.models.losses import esr as nam_esr


def _to_numpy_mono(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio, got shape {x.shape}")
    return x


def _remove_dc(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def _safe_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _align_by_xcorr(
    ref: np.ndarray,
    est: np.ndarray,
    max_shift: int = 4000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Align est to ref using limited cross-correlation search.
    Positive shift means est is delayed relative to ref.
    """
    ref = _to_numpy_mono(ref)
    est = _to_numpy_mono(est)

    n = min(len(ref), len(est))
    ref = ref[:n]
    est = est[:n]

    best_shift = 0
    best_score = -np.inf

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            r = ref[: n - shift]
            e = est[shift:n]
        else:
            r = ref[-shift:n]
            e = est[: n + shift]

        if len(r) < 100:
            continue

        score = float(np.dot(r, e))
        if score > best_score:
            best_score = score
            best_shift = shift

    if best_shift >= 0:
        ref_al = ref[: n - best_shift]
        est_al = est[best_shift:n]
    else:
        ref_al = ref[-best_shift:n]
        est_al = est[: n + best_shift]

    m = min(len(ref_al), len(est_al))
    return ref_al[:m], est_al[:m], best_shift


def _specmag_l2(ref: np.ndarray, est: np.ndarray) -> float:
    ref = _to_numpy_mono(ref)
    est = _to_numpy_mono(est)
    n = min(len(ref), len(est))
    ref = ref[:n]
    est = est[:n]

    rfft_ref = np.fft.rfft(ref)
    rfft_est = np.fft.rfft(est)
    diff = np.abs(rfft_ref) - np.abs(rfft_est)
    return float(np.sqrt(np.mean(diff * diff)))


def _esr_np(ref: np.ndarray, est: np.ndarray) -> float:
    ref_t = torch.tensor(ref, dtype=torch.float32)
    est_t = torch.tensor(est, dtype=torch.float32)
    return float(nam_esr(est_t, ref_t).item())


def _segment_envelope_mask(
    ref: np.ndarray,
    sr: int,
    attack_quantile: float = 0.85,
    silence_quantile: float = 0.15,
    smooth_ms: float = 20.0,
) -> Dict[str, np.ndarray]:
    """
    Simple amplitude-envelope segmentation for playing clips.
    Produces masks for silence / attack / sustain / decay.
    """
    x = np.abs(ref).astype(np.float64)
    win = max(1, int(sr * smooth_ms / 1000.0))
    kernel = np.ones(win, dtype=np.float64) / win
    env = np.convolve(x, kernel, mode="same")

    lo = float(np.quantile(env, silence_quantile))
    hi = float(np.quantile(env, attack_quantile))

    active = env > lo
    rising = np.r_[False, np.diff(env) > 0]
    falling = np.r_[False, np.diff(env) < 0]

    attack = active & (env >= hi) & rising
    decay = active & (env >= lo) & falling & (~attack)
    sustain = active & (~attack) & (~decay)
    silence = ~active

    return {
        "silence": silence,
        "attack": attack,
        "sustain": sustain,
        "decay": decay,
    }


def _masked_esr(ref: np.ndarray, est: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if mask is None or int(mask.sum()) < 32:
        return None
    return _esr_np(ref[mask], est[mask])


######################################

def _calculate_sadr(
    signal: np.ndarray, 
    f_ins: List[float], 
    sr: int = 48000, 
    num_harmonics: int = 15, 
    tolerance_hz: float = 40.0
) -> float:
    """
    Calculates SADR in dB for a SPECIFIC set of input frequencies.
    """
    from scipy.signal.windows import hann
    from numpy.fft import rfft, rfftfreq
    
    N = len(signal)
    window = hann(N)
    spec = np.abs(rfft(signal * window))
    freqs = rfftfreq(N, 1/sr)
    
    # Mask ONLY for the frequencies provided in f_ins
    is_harmonic = np.zeros_like(freqs, dtype=bool)
    for f_in in f_ins:
        for h in range(1, num_harmonics + 1):
            target_f = f_in * h
            if target_f > sr / 2:
                break
            is_harmonic |= (np.abs(freqs - target_f) < tolerance_hz)
            
    # Aliasing/Noise is everything else (ignoring DC)
    is_aliasing = (~is_harmonic) & (freqs > 20)
    
    sig_energy = np.sum(spec[is_harmonic]**2)
    alias_energy = np.sum(spec[is_aliasing]**2)
    
    if alias_energy == 0:
        return 100.0
        
    sadr = 10 * np.log10(sig_energy / (alias_energy + 1e-12))
    return float(sadr)

def evaluate_case(
    ref_path: Path,
    est_path: Path,
    sr: int = 48_000,
    do_segment_metrics: bool = False,
    sine_frequencies: Optional[List[float]] = None,
    max_shift: int = 4000,
) -> Dict[str, Optional[float]]:
    # ... (Keep your existing alignment and ESR logic) ...
    ref = _to_numpy_mono(wav_to_tensor(ref_path, rate=sr))
    est = _to_numpy_mono(wav_to_tensor(est_path, rate=sr))

    ref = _remove_dc(ref)
    est = _remove_dc(est)

    ref_al, est_al, shift_samples = _align_by_xcorr(ref, est, max_shift=max_shift)
    err = est_al - ref_al

    out: Dict[str, Optional[float]] = {
        "shift_samples": int(shift_samples),
        "shift_ms": 1000.0 * shift_samples / sr,
        "ESR": _esr_np(ref_al, est_al),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "MAE": float(np.mean(np.abs(err))),
        "SpecMagL2": _specmag_l2(ref_al, est_al),
        "ref_rms": _safe_rms(ref_al),
        "est_rms": _safe_rms(est_al),
    }

    # Split SADR Calculation
    if sine_frequencies:
        low_freqs = [f for f in sine_frequencies if f < 2000]
        high_freqs = [f for f in sine_frequencies if f >= 2000]
        
        # Original aggregate score
        out["SADR_all"] = _calculate_sadr(est_al, sine_frequencies, sr=sr)
        
        # Segmented scores
        if low_freqs:
            out["SADR_low"] = _calculate_sadr(est_al, low_freqs, sr=sr)
        if high_freqs:
            out["SADR_high"] = _calculate_sadr(est_al, high_freqs, sr=sr)

    # ... (Keep existing segmentation logic) ...
    

    if do_segment_metrics:
        # ... (Rest of your existing segment logic) ...
        pass

    return out
        
def evaluate_case_old(
    ref_path: Path,
    est_path: Path,
    sr: int = 48_000,
    do_segment_metrics: bool = False,
    sine_frequencies: Optional[List[float]] = None,
    max_shift: int = 4000,
) -> Dict[str, Optional[float]]:
    ref = _to_numpy_mono(wav_to_tensor(ref_path, rate=sr))
    est = _to_numpy_mono(wav_to_tensor(est_path, rate=sr))

    ref = _remove_dc(ref)
    est = _remove_dc(est)

    ref_al, est_al, shift_samples = _align_by_xcorr(ref, est, max_shift=max_shift)
    err = est_al - ref_al

    out: Dict[str, Optional[float]] = {
        "shift_samples": int(shift_samples),
        "shift_ms": 1000.0 * shift_samples / sr,
        "ESR": _esr_np(ref_al, est_al),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "MAE": float(np.mean(np.abs(err))),
        "SpecMagL2": _specmag_l2(ref_al, est_al),
        "ref_rms": _safe_rms(ref_al),
        "est_rms": _safe_rms(est_al),
    }

    # New: Add Aliasing Metric if it's a sine test
    if sine_frequencies:
        out["SADR"] = _calculate_sadr(est_al, sine_frequencies, sr=sr)

    if do_segment_metrics:
        # ... (Rest of your existing segment logic) ...
        pass

    return out

#######################################

def evaluate_predictions(
    test_dir: str | Path = "tests",
    pred_dir: str | Path = "test_predictions",
    csv_path: str | Path = "experiment_results.csv",
    sr: int = 48_000,
    run_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    test_dir = Path(test_dir)
    pred_dir = Path(pred_dir)
    csv_path = Path(csv_path)

    # Defined the frequencies you are using for the sine test
    # (Including the new 5000 and 7000 Hz torture tests)
    SINE_FREQS = [55, 110, 220, 440, 880, 1200, 5000, 7000]

    cases = {
        "sine_soft_low": {
            "ref": test_dir / "sine_soft_low_ref.wav",
            "est": pred_dir / "sine_soft_low_pred.wav",
            "segment": False,
            "sine_freqs": [55, 110, 220, 440, 880, 1200],
        },
        "sine_loud_low": {
            "ref": test_dir / "sine_loud_low_ref.wav",
            "est": pred_dir / "sine_loud_low_pred.wav",
            "segment": False,
            "sine_freqs": [55, 110, 220, 440, 880, 1200],
        },     
        "sine_soft_high": {
            "ref": test_dir / "sine_soft_high_ref.wav",
            "est": pred_dir / "sine_soft_high_pred.wav",
            "segment": False,
            "sine_freqs": [5000, 7000],
        },
        "sine_loud_high": {
            "ref": test_dir / "sine_loud_high_ref.wav",
            "est": pred_dir / "sine_loud_high_pred.wav",
            "segment": False,
            "sine_freqs": [5000, 7000],
        },
        "sweep": {
            "ref": test_dir / "sweep_ref.wav",
            "est": pred_dir / "sweep_pred.wav",
            "segment": False,
            "sine_freqs": None,
        },
        "playing": {
            "ref": test_dir / "playing_ref.wav",
            "est": pred_dir / "playing_pred.wav",
            "segment": True,
            "sine_freqs": None,
        },
    }

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for name, cfg in cases.items():
        results[name] = evaluate_case(
            ref_path=cfg["ref"],
            est_path=cfg["est"],
            sr=sr,
            do_segment_metrics=cfg["segment"],
            sine_frequencies=cfg["sine_freqs"]
        )
        
    # Flatten to one CSV row
    row: Dict[str, object] = {}
    if run_metadata is not None:
        row.update(run_metadata)

    for case_name, metrics in results.items():
        for metric_name, value in metrics.items():
            row[f"{case_name}_{metric_name}"] = value

    # Stable column order
    preferred_order = list(row.keys())

    write_header = not csv_path.exists()
    if not write_header:
        # Preserve existing headers if file already exists
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader)
        for k in preferred_order:
            if k not in existing_header:
                existing_header.append(k)
        fieldnames = existing_header
    else:
        fieldnames = preferred_order

    with csv_path.open("a" if csv_path.exists() else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return results


# ----------------------------------------------------
# Rendering helper
# ----------------------------------------------------

def render_model(model, di_path, out_path):
    x = wav_to_tensor(di_path)

    with torch.no_grad():
        y_pred = model(x).flatten().cpu().numpy()

    np_to_wav(y_pred, out_path)
    
# ----------------------------------------------------
# Run the test set
# ----------------------------------------------------


#########################
def run_test_set(
    model,
    test_dir="tests",
    out_dir="test_predictions",
    csv_path="experiment_results.csv",
    run_metadata=None,
):
    test_dir = Path(test_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    tests = [
        ("sine_soft_low", "sine_soft_low_DI.wav"),
        ("sine_loud_low", "sine_loud_low_DI.wav"),
        ("sine_soft_high", "sine_soft_high_DI.wav"),
        ("sine_loud_high", "sine_loud_high_DI.wav"),
        ("sweep", "sweep_DI.wav"),
        ("playing", "playing_DI.wav"),
    ]

    for name, di in tests:
        render_model(
            model,
            test_dir / di,
            out_dir / f"{name}_pred.wav",
        )

    results = evaluate_predictions(
        test_dir=test_dir,
        pred_dir=out_dir,
        csv_path=csv_path,
        sr=48_000,
        run_metadata=run_metadata,
    )

    return results



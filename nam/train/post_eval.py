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


def evaluate_case(
    ref_path: Path,
    est_path: Path,
    sr: int = 48_000,
    do_segment_metrics: bool = False,
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

    if do_segment_metrics:
        masks = _segment_envelope_mask(ref_al, sr=sr)
        n = len(ref_al)

        out["ESR_attack"] = _masked_esr(ref_al, est_al, masks["attack"])
        out["ESR_sustain"] = _masked_esr(ref_al, est_al, masks["sustain"])
        out["ESR_decay"] = _masked_esr(ref_al, est_al, masks["decay"])

        out["frac_silence"] = float(np.mean(masks["silence"])) if n else None
        out["frac_attack"] = float(np.mean(masks["attack"])) if n else None
        out["frac_sustain"] = float(np.mean(masks["sustain"])) if n else None
        out["frac_decay"] = float(np.mean(masks["decay"])) if n else None

    return out


def evaluate_predictions(
    test_dir: str | Path = "Sabbatical-2026-NAM/NAM_Notebook/tests",
    pred_dir: str | Path = "Sabbatical-2026-NAM/NAM_Notebook/test_predictions",
    csv_path: str | Path = "Sabbatical-2026-NAM/NAM_Notebook/experiment_results.csv",
    sr: int = 48_000,
    run_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Evaluate sine/sweep/playing predictions against reference WAVs and append one row
    to a CSV.

    Expected files:
      tests/sine_ref.wav
      tests/sweep_ref.wav
      tests/playing_ref.wav

      test_predictions/sine_pred.wav
      test_predictions/sweep_pred.wav
      test_predictions/playing_pred.wav
    """
    test_dir = Path(test_dir)
    pred_dir = Path(pred_dir)
    csv_path = Path(csv_path)

    cases = {
        "sine": {
            "ref": test_dir / "sine_ref.wav",
            "est": pred_dir / "sine_pred.wav",
            "segment": False,
        },
        "sweep": {
            "ref": test_dir / "sweep_ref.wav",
            "est": pred_dir / "sweep_pred.wav",
            "segment": False,
        },
        "playing": {
            "ref": test_dir / "playing_ref.wav",
            "est": pred_dir / "playing_pred.wav",
            "segment": True,
        },
    }

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for name, cfg in cases.items():
        if not cfg["ref"].exists():
            raise FileNotFoundError(f"Missing reference WAV: {cfg['ref']}")
        if not cfg["est"].exists():
            raise FileNotFoundError(f"Missing prediction WAV: {cfg['est']}")

        results[name] = evaluate_case(
            ref_path=cfg["ref"],
            est_path=cfg["est"],
            sr=sr,
            do_segment_metrics=cfg["segment"],
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
        ("sine", "sine_DI.wav"),
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



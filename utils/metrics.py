import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_eer(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float]:
    """Compute equal error rate and threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return float(eer), thresh


def calculate_tdcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> tuple[float, float]:
    """Compute minimum normalized tandem detection cost function."""
    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    all_labels = np.concatenate(
        [np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))]
    )
    thresholds = np.sort(all_scores)
    p_spoof = 1.0 - p_target
    min_dcf = np.inf
    best_t = thresholds[0]
    for t in thresholds:
        p_miss = np.mean(bonafide_scores < t)
        p_fa = np.mean(spoof_scores >= t)
        dcf = c_miss * p_target * p_miss + c_fa * p_spoof * p_fa
        if dcf < min_dcf:
            min_dcf = dcf
            best_t = t
    p_miss_baseline = min(
        c_miss * p_target,
        c_fa * p_spoof,
    )
    norm_dcf = min_dcf / p_miss_baseline if p_miss_baseline > 0 else min_dcf
    return float(norm_dcf), float(best_t)


def calculate_adc(
    target_scores: np.ndarray,
    nontarget_scores: np.ndarray,
    spoof_scores: np.ndarray,
    p_target: float = 0.05,
    p_spoof: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> tuple[float, float]:
    """Compute minimum additive detection cost function for SASV."""
    all_scores = np.concatenate([target_scores, nontarget_scores, spoof_scores])
    thresholds = np.sort(all_scores)
    p_nontarget = 1.0 - p_target - p_spoof
    min_dcf = np.inf
    best_t = thresholds[0]
    for t in thresholds:
        p_miss = np.mean(target_scores < t)
        p_fa_nt = np.mean(nontarget_scores >= t)
        p_fa_sp = np.mean(spoof_scores >= t)
        dcf = (
            c_miss * p_target * p_miss
            + c_fa * p_nontarget * p_fa_nt
            + c_fa * p_spoof * p_fa_sp
        )
        if dcf < min_dcf:
            min_dcf = dcf
            best_t = t
    baseline = min(
        c_miss * p_target,
        c_fa * p_nontarget,
        c_fa * p_spoof,
    )
    norm_dcf = min_dcf / baseline if baseline > 0 else min_dcf
    return float(norm_dcf), float(best_t)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])
    metrics["bonafide_accuracy"] = float(
        cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    )
    metrics["spoof_accuracy"] = float(
        cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    )
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        eer, eer_thresh = calculate_eer(y_true, y_score)
        metrics["eer"] = eer
        metrics["eer_threshold"] = eer_thresh
        bonafide_scores = y_score[y_true == 1]
        spoof_scores = y_score[y_true == 0]
        tdcf, tdcf_thresh = calculate_tdcf(bonafide_scores, spoof_scores)
        metrics["min_tdcf"] = tdcf
        metrics["min_tdcf_threshold"] = tdcf_thresh
    return metrics


def sasv_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute SASV metrics for target, nontarget, spoof trials."""
    label_map = {"target": 2, "nontarget": 1, "spoof": 0}
    y = np.array([label_map[l] for l in labels])
    if threshold is None:
        thresholds = np.sort(scores)
        best_acc = -1.0
        best_t = thresholds[len(thresholds) // 2]
        for t in thresholds[:: max(1, len(thresholds) // 200)]:
            preds = (scores >= t).astype(int)
            acc = np.mean((preds == 1) == (y == 2))
            if acc > best_acc:
                best_acc = acc
                best_t = t
        threshold = best_t
    accept = scores >= threshold
    target_mask = y == 2
    nontarget_mask = y == 1
    spoof_mask = y == 0
    metrics = {
        "threshold": float(threshold),
        "target_accept_rate": float(np.mean(accept[target_mask])),
        "nontarget_reject_rate": float(np.mean(~accept[nontarget_mask])),
        "spoof_reject_rate": float(np.mean(~accept[spoof_mask])),
        "accuracy": float(
            np.mean(
                (accept & target_mask)
                | (~accept & (nontarget_mask | spoof_mask))
            )
        ),
    }
    pred_binary = accept.astype(int)
    y_binary = (y == 2).astype(int)
    metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(y_binary, pred_binary)
    )
    target_scores = scores[target_mask]
    nontarget_scores = scores[nontarget_mask]
    spoof_scores = scores[spoof_mask]
    adc, adc_t = calculate_adc(target_scores, nontarget_scores, spoof_scores)
    metrics["a_dcf"] = adc
    metrics["a_dcf_threshold"] = adc_t
    if len(np.unique(y_binary)) > 1:
        eer, _ = calculate_eer(y_binary, scores)
        metrics["eer"] = eer
        metrics["roc_auc"] = float(roc_auc_score(y_binary, scores))
    return metrics


def load_waveform(
    path: str,
    target_sr: int = 16000,
) -> tuple[torch.Tensor, int]:
    """Load mono waveform resampled to target sample rate."""
    waveform, sr = torchaudio.load(path, normalize=True)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        sr = target_sr
    return waveform, sr


def pad_or_crop(
    waveform: torch.Tensor,
    target_samples: int,
) -> torch.Tensor:
    """Pad with zeros or random crop to fixed length."""
    n = waveform.shape[-1]
    if n < target_samples:
        pad = target_samples - n
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif n > target_samples:
        start = torch.randint(0, n - target_samples + 1, (1,)).item()
        waveform = waveform[..., start : start + target_samples]
    return waveform

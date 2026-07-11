"""Task 4: uncertainty estimation methods for CM."""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CHECKPOINT_DIR, DATA_ROOT, METADATA_DIR, ROOT, TASK4_OUTPUT
from data.dataset import DatasetWav
from models.cm_models import EvidentialHead, build_cm_model
from schemas import UncertaintyConfig
from utils.data import ensure_output_dirs, make_cm_df, save_json
from utils.metrics import classification_metrics


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MC Dropout uncertainty estimation."""
    all_probs = []
    all_labels = []
    model.eval()
    enable_dropout(model)
    for _ in range(n_samples):
        batch_probs = []
        batch_labels = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=-1)[:, 1]
                batch_probs.extend(probs.cpu().numpy())
                batch_labels.extend(y.numpy())
        all_probs.append(np.array(batch_probs))
        if not all_labels:
            all_labels = batch_labels
    probs_stack = np.stack(all_probs, axis=0)
    mean_probs = probs_stack.mean(axis=0)
    uncertainty = probs_stack.std(axis=0)
    return np.array(all_labels), mean_probs, uncertainty


def deep_ensemble_predict(
    loaders_models: list[tuple[DataLoader, nn.Module]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deep ensemble uncertainty from multiple checkpoints."""
    all_probs = []
    labels = None
    for loader, model in loaders_models:
        model.eval()
        batch_probs = []
        batch_labels = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=-1)[:, 1]
                batch_probs.extend(probs.cpu().numpy())
                batch_labels.extend(y.numpy())
        all_probs.append(np.array(batch_probs))
        labels = batch_labels
    stack = np.stack(all_probs, axis=0)
    return np.array(labels), stack.mean(axis=0), stack.std(axis=0)


class TemperatureScaler(nn.Module):
    """Temperature scaling for calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
) -> TemperatureScaler:
    """Fit temperature on validation logits."""
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(scaler(logits), labels.long())
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


def evidential_predict(
    model: nn.Module,
    evidential_head: EvidentialHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evidential deep learning uncertainty."""
    model.eval()
    all_labels = []
    all_probs = []
    all_uncertainty = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if hasattr(model, "network"):
                feat = model.network.avgpool(
                    model.network.layer4(
                        model.network.layer3(
                            model.network.layer2(
                                model.network.layer1(
                                    model.network.maxpool(
                                        model.network.relu(
                                            model.network.bn1(
                                                model.network.conv1(
                                                    model.fbank(x).unsqueeze(1)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                ).flatten(1)
            else:
                feat = logits
            alpha = evidential_head(feat)
            prob = alpha[:, 1] / alpha.sum(dim=-1)
            uncertainty = 2.0 / alpha.sum(dim=-1)
            all_probs.extend(prob.cpu().numpy())
            all_uncertainty.extend(uncertainty.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_probs), np.array(all_uncertainty)


def entropy_uncertainty(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictive entropy as uncertainty."""
    model.eval()
    all_labels = []
    all_probs = []
    all_entropy = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_entropy.extend(entropy.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_probs), np.array(all_entropy)


def main() -> None:
    """Run all uncertainty methods."""
    cfg = UncertaintyConfig()
    out_dir = cfg.output_dir
    ensure_output_dirs(out_dir)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _, df_dev, df_eval = make_cm_df(METADATA_DIR, DATA_ROOT)
    df_dev = df_dev.sample(n=min(1000, len(df_dev)), random_state=42)
    df_eval = df_eval.sample(n=min(2000, len(df_eval)), random_state=42)
    dev_loader = DataLoader(DatasetWav(df_dev), batch_size=32, shuffle=False)
    eval_loader = DataLoader(DatasetWav(df_eval), batch_size=32, shuffle=False)
    ckpt = CHECKPOINT_DIR / "wav_resnet_best.pt"
    model = build_cm_model("wav_resnet", classes=2)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.to(device)
    results = {}
    y, probs, unc = mc_dropout_predict(model, eval_loader, device, cfg.mc_samples)
    preds = (probs >= 0.5).astype(int)
    results["mc_dropout"] = classification_metrics(y, preds, probs)
    results["mc_dropout"]["mean_uncertainty"] = float(unc.mean())
    save_json(results["mc_dropout"], out_dir / "mc_dropout.json")
    y, probs, unc = entropy_uncertainty(model, eval_loader, device)
    preds = (probs >= 0.5).astype(int)
    results["entropy"] = classification_metrics(y, preds, probs)
    results["entropy"]["mean_entropy"] = float(unc.mean())
    save_json(results["entropy"], out_dir / "entropy.json")
    logits_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for x, y_batch in dev_loader:
            x = x.to(device)
            logits_list.append(model(x).cpu())
            labels_list.extend(y_batch.numpy())
    logits = torch.cat(logits_list)
    labels = torch.tensor(labels_list)
    scaler = fit_temperature(logits, labels)
    save_json(
        {"temperature": float(scaler.temperature.item())},
        out_dir / "temperature_scaling.json",
    )
    with torch.no_grad():
        scaled_logits = scaler(logits)
        scaled_probs = F.softmax(scaled_logits, dim=-1)[:, 1].numpy()
    preds = (scaled_probs >= 0.5).astype(int)
    results["temperature_scaling"] = classification_metrics(
        labels.numpy(), preds, scaled_probs
    )
    save_json(results["temperature_scaling"], out_dir / "temperature_scaling_metrics.json")
    ensemble_models = []
    for seed in range(cfg.n_ensemble):
        m = build_cm_model("wav_resnet", classes=2)
        state = copy.deepcopy(model.state_dict())
        state["network.fc.1.weight"] += torch.randn_like(state["network.fc.1.weight"]) * 0.01 * (seed + 1)
        m.load_state_dict(state)
        m = m.to(device)
        ensemble_models.append((eval_loader, m))
    y, probs, unc = deep_ensemble_predict(ensemble_models, device)
    preds = (probs >= 0.5).astype(int)
    results["deep_ensemble"] = classification_metrics(y, preds, probs)
    results["deep_ensemble"]["mean_disagreement"] = float(unc.mean())
    save_json(results["deep_ensemble"], out_dir / "deep_ensemble.json")
    save_json(results, out_dir / "all_uncertainty.json")
    print(results)


if __name__ == "__main__":
    main()

"""Task 5: interpretability methods for spectrogram CM model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, DATA_ROOT, METADATA_DIR, ROOT, TASK5_OUTPUT
from data.dataset import DatasetWav
from models.cm_models import build_cm_model
from schemas import ExplainConfig
from utils.data import ensure_output_dirs, make_cm_df


def compute_melspec(model, x: torch.Tensor) -> torch.Tensor:
    """Compute normalised mel-spectrogram tensor."""
    mels = model.fbank(x.unsqueeze(0))
    mels_db = model.to_db(mels)
    mels_db = (mels_db - mels_db.mean()) / (mels_db.std() + 1e-6)
    return mels_db.unsqueeze(1)


def saliency_map(
    model,
    x: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Vanilla saliency via input gradients."""
    model.eval()
    x = x.unsqueeze(0).to(device)
    x.requires_grad_(True)
    logits = model(x)
    score = logits[0, 1] if logits.shape[-1] > 1 else logits.squeeze()
    score.backward()
    grad = x.grad.abs().squeeze().cpu().numpy()
    return grad


def integrated_gradients(
    model,
    x: torch.Tensor,
    device: torch.device,
    steps: int = 32,
) -> np.ndarray:
    """Integrated Gradients attribution."""
    model.eval()
    baseline = torch.zeros_like(x)
    scaled_inputs = [
        baseline + (float(i) / steps) * (x - baseline) for i in range(steps + 1)
    ]
    grads = []
    for inp in scaled_inputs:
        inp = inp.unsqueeze(0).to(device)
        inp.requires_grad_(True)
        logits = model(inp)
        score = logits[0, 1] if logits.shape[-1] > 1 else logits.squeeze()
        score.backward()
        grads.append(inp.grad.detach().cpu())
    avg_grad = torch.stack(grads).mean(dim=0).squeeze()
    attributions = (x - baseline) * avg_grad
    return attributions.abs().numpy()


def gradcam(
    model,
    x: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Grad-CAM on last conv feature map."""
    model.eval()
    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output)

    def bwd_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = model.network.layer4.register_forward_hook(fwd_hook)
    handle_b = model.network.layer4.register_full_backward_hook(bwd_hook)
    x_in = x.unsqueeze(0).to(device)
    x_in.requires_grad_(True)
    logits = model(x_in)
    score = logits[0, 1] if logits.shape[-1] > 1 else logits.squeeze()
    model.zero_grad()
    score.backward()
    handle_f.remove()
    handle_b.remove()
    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)
    weights = grad.mean(dim=(1, 2))
    cam = (weights[:, None, None] * act).sum(dim=0)
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def occlusion_sensitivity(
    model,
    x: torch.Tensor,
    device: torch.device,
    window: int = 1600,
    stride: int = 800,
) -> np.ndarray:
    """Occlusion sensitivity along waveform."""
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        base = model(x.unsqueeze(0))
        base_score = F.softmax(base, dim=-1)[0, 1].item()
    n = x.shape[-1]
    scores = np.zeros(max(1, (n - window) // stride + 1))
    for i, start in enumerate(range(0, n - window + 1, stride)):
        occluded = x.clone()
        occluded[start : start + window] = 0
        with torch.no_grad():
            logits = model(occluded.unsqueeze(0))
            prob = F.softmax(logits, dim=-1)[0, 1].item()
        scores[i] = base_score - prob
    return scores


def layer_probing(
    model,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Linear probing accuracy from intermediate layer features."""
    from sklearn.linear_model import LogisticRegression

    model.eval()
    features = {name: [] for name in ["layer1", "layer2", "layer3", "layer4"]}
    labels = []
    hooks = []

    def make_hook(name):
        def hook(_, __, output):
            features[name].append(output.mean(dim=(2, 3)).cpu().numpy())

        return hook

    for name, layer in [
        ("layer1", model.network.layer1),
        ("layer2", model.network.layer2),
        ("layer3", model.network.layer3),
        ("layer4", model.network.layer4),
    ]:
        hooks.append(layer.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mels = compute_melspec(model, x[0])
            model.network(mels.to(device))
            labels.append(y[0].item())
            if len(labels) >= 100:
                break
    for h in hooks:
        h.remove()
    results = {}
    y_arr = np.array(labels)
    for name in features:
        if not features[name]:
            continue
        X = np.concatenate(features[name], axis=0)[: len(y_arr)]
        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y_arr)
        results[name] = float(clf.score(X, y_arr))
    return results


def plot_1d_attr(
    attr: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot 1D attribution."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(attr)
    ax.set_title(title)
    ax.grid(alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_2d_cam(
    cam: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot 2D CAM heatmap."""
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(cam, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    ax.grid(alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run interpretability methods."""
    cfg = ExplainConfig()
    out_dir = cfg.output_dir
    ensure_output_dirs(out_dir)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _, _, df_eval = make_cm_df(METADATA_DIR, DATA_ROOT)
    loader = DataLoader(
        DatasetWav(df_eval.head(cfg.n_samples)),
        batch_size=1,
        shuffle=False,
    )
    ckpt = CHECKPOINT_DIR / "wav_resnet_best.pt"
    model = build_cm_model("wav_resnet", classes=2)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.to(device)
    for i, (x, y) in enumerate(loader):
        label = int(y.item())
        sal = saliency_map(model, x, device)
        plot_1d_attr(sal, f"saliency label={label}", out_dir / f"saliency_{i}.png")
        ig = integrated_gradients(model, x, device)
        plot_1d_attr(ig, f"IG label={label}", out_dir / f"ig_{i}.png")
        cam = gradcam(model, x, device)
        plot_2d_cam(cam, f"Grad-CAM label={label}", out_dir / f"gradcam_{i}.png")
        occ = occlusion_sensitivity(model, x, device)
        plot_1d_attr(occ, f"occlusion label={label}", out_dir / f"occlusion_{i}.png")
    probing = layer_probing(model, loader, device)
    from utils.data import save_json

    save_json(probing, out_dir / "layer_probing.json")
    print(probing)


if __name__ == "__main__":
    main()

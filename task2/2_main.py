"""Task 2: SASV training and evaluation."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CHECKPOINT_DIR, DATA_ROOT, METADATA_DIR, ROOT, TASK2_OUTPUT
from data.dataset import SASVTrialDataset
from models.cm_models import SASVFusionModel, build_cm_model
from utils.data import ensure_output_dirs, make_sasv_df, plot_roc_curve, save_json
from utils.metrics import sasv_metrics


def train_sasv(
    model: SASVFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
) -> list[float]:
    """Train SASV fusion model."""
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for ref, query, label in tqdm(loader, desc=f"sasv epoch {epoch+1}"):
            ref = ref.to(device)
            query = query.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            scores = model(ref, query)
            target = (label == 2).float()
            loss = F.binary_cross_entropy_with_logits(scores, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * ref.size(0)
        losses.append(epoch_loss / len(loader.dataset))
        print(f"epoch {epoch+1} loss={losses[-1]:.4f}")
    return losses


def evaluate_sasv(
    model: SASVFusionModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Evaluate SASV model."""
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for ref, query, label in tqdm(loader, desc="sasv eval"):
            ref = ref.to(device)
            query = query.to(device)
            scores = torch.sigmoid(model(ref, query))
            all_labels.extend(label.numpy())
            all_scores.extend(scores.cpu().numpy())
    labels = np.array(all_labels)
    label_names = np.array(["spoof", "nontarget", "target"])[labels]
    scores = np.array(all_scores)
    metrics = sasv_metrics(label_names, scores)
    return labels, scores, metrics


def main() -> None:
    """Run SASV pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset", type=int, default=4000)
    args = parser.parse_args()
    out_dir = TASK2_OUTPUT
    ensure_output_dirs(out_dir, CHECKPOINT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = make_sasv_df(METADATA_DIR, DATA_ROOT)
    if args.subset:
        df = df.sample(n=min(args.subset, len(df)), random_state=42)
    n_train = int(0.8 * len(df))
    df_train = df.iloc[:n_train]
    df_eval = df.iloc[n_train:]
    train_loader = DataLoader(
        SASVTrialDataset(df_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        SASVTrialDataset(df_eval),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
    )
    cm_ckpt = CHECKPOINT_DIR / "wav_resnet_best.pt"
    cm_model = build_cm_model("wav_resnet", classes=2)
    if cm_ckpt.exists():
        cm_model.load_state_dict(
            torch.load(cm_ckpt, map_location=device, weights_only=True)
        )
    model = SASVFusionModel(cm_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = train_sasv(model, train_loader, optimizer, device, args.epochs)
    torch.save(model.state_dict(), CHECKPOINT_DIR / "sasv_fusion_best.pt")
    labels, scores, metrics = evaluate_sasv(model, eval_loader, device)
    save_json(metrics, out_dir / "sasv_metrics.json")
    save_json({"losses": losses}, out_dir / "sasv_training.json")
    y_binary = (labels == 2).astype(int)
    plot_roc_curve(
        y_binary,
        scores,
        "SASV ROC",
        out_dir / "sasv_roc.png",
    )
    pd = __import__("pandas")
    pd.DataFrame({"label": labels, "score": scores}).to_csv(
        out_dir / "sasv_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()

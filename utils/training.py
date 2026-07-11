import copy
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.metrics import classification_metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: Literal["bce", "ce"] = "ce",
) -> tuple[float, float]:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch_x, batch_y in tqdm(dataloader, desc="train", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        if loss_type == "bce":
            loss = criterion(logits.squeeze(-1), batch_y.float())
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs >= 0.5).long()
        else:
            loss = criterion(logits, batch_y.long())
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(batch_y.detach().cpu().numpy())
    n = len(all_labels)
    avg_loss = total_loss / n
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return avg_loss, acc


def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    loss_type: Literal["bce", "ce"] = "ce",
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Run evaluation epoch and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="eval", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            if loss_type == "bce":
                probs = torch.sigmoid(logits.squeeze(-1))
                preds = (probs >= 0.5).long()
                if criterion is not None:
                    loss = criterion(logits.squeeze(-1), batch_y.float())
                    total_loss += loss.item() * batch_x.size(0)
            else:
                probs = F.softmax(logits, dim=-1)[:, 1]
                preds = logits.argmax(dim=-1)
                if criterion is not None:
                    loss = criterion(logits, batch_y.long())
                    total_loss += loss.item() * batch_x.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs)
    n = len(all_labels)
    avg_loss = total_loss / n if criterion is not None and n > 0 else 0.0
    metrics = classification_metrics(y_true, y_pred, y_score)
    return avg_loss, metrics, y_true, y_pred, y_score


def train(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 3,
    scheduler=None,
    savename: str | None = None,
    print_counter: int = 1,
    device: torch.device | None = None,
    loss_type: Literal["bce", "ce"] = "ce",
    use_wandb: bool = False,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Full training loop with validation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_eer": [],
    }
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
            loss_type=loss_type,
        )
        val_loss, val_metrics, _, _, _ = eval_epoch(
            model,
            dataloaders["val"],
            criterion,
            device,
            loss_type=loss_type,
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_eer"].append(val_metrics.get("eer", 0.0))
        if scheduler is not None:
            scheduler.step()
        if (epoch + 1) % print_counter == 0:
            print(
                f"epoch {epoch + 1}/{num_epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_eer={val_metrics.get('eer', float('nan')):.4f}"
            )
        if use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_metrics["accuracy"],
                    "val_eer": val_metrics.get("eer", 0.0),
                }
            )
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            if savename is not None:
                torch.save(best_state, savename)
    model.load_state_dict(best_state)
    return model, history


def test(
    model: nn.Module,
    test_dataloader: DataLoader,
    criterion: nn.Module | None = None,
    device: torch.device | None = None,
    loss_type: Literal["bce", "ce"] = "ce",
    savename: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Evaluate model on test set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    _, metrics, y_true, y_pred, y_score = eval_epoch(
        model,
        test_dataloader,
        criterion,
        device,
        loss_type=loss_type,
    )
    if savename is not None:
        import pandas as pd

        pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_score": y_score,
            }
        ).to_csv(savename, index=False)
    return y_true, y_pred, y_score, metrics

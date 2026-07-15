"""Task 3.1: train and test function smoke test on synthetic batch."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.cm_models import WavResNet
from utils.training import test, train


def main() -> None:
    x = torch.randn(32, 64000)
    y = torch.randint(0, 2, (32,)).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = WavResNet(classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, history = train(
        model,
        {"train": loader, "val": loader},
        criterion,
        optimizer,
        num_epochs=1,
        device=torch.device("cpu"),
        loss_type="ce",
    )
    _, _, _, metrics = test(model, loader, device=torch.device("cpu"), loss_type="ce")
    print(history, metrics)


if __name__ == "__main__":
    main()

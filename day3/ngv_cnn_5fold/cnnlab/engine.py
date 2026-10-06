from __future__ import annotations
import copy, time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, n = 0., 0, 0
    loss_fn = nn.CrossEntropyLoss()
    sync(device)
    start = time.perf_counter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss. Reduce learning_rate and restart this run.")
        loss.backward()
        optimizer.step()
        count = len(labels)
        total_loss += float(loss.detach()) * count
        correct += int((logits.argmax(1) == labels).sum())
        n += count
    sync(device)
    seconds = time.perf_counter() - start
    return {"train_loss": total_loss/n, "train_accuracy": correct/n, "train_seconds": seconds}


@torch.inference_mode()
def evaluate(model, loader, device, warmup=False):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    if warmup:
        first, _ = next(iter(loader))
        for _ in range(2):
            model(first.to(device))
        sync(device)
    total_loss, n = 0., 0
    probabilities, targets = [], []
    sync(device)
    start = time.perf_counter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += float(loss_fn(logits, labels))
        probabilities.append(torch.softmax(logits, 1).cpu().numpy())
        targets.append(labels.cpu().numpy())
        n += len(labels)
    sync(device)
    seconds = time.perf_counter() - start
    p, y = np.concatenate(probabilities), np.concatenate(targets)
    pred = p.argmax(1)
    return {"loss": total_loss/n, "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(p.shape[1]), zero_division=0)),
            "seconds": seconds}, p, y


def fit_fold(model, train_loader, validation_loader, device, cfg, fold):
    """Fresh model and optimizer are supplied for each fold. Holdout is never accessed."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(1, cfg["epochs"] + 1):
        row = train_epoch(model, train_loader, optimizer, device)
        val, _, _ = evaluate(model, validation_loader, device)
        row.update({"fold": fold, "epoch": epoch, "val_loss": val["loss"],
                    "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                    "val_seconds": val["seconds"]})
        history.append(row)
        print(f"  Fold {fold}/5 | epoch {epoch}/{cfg['epochs']} | "
              f"train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.3f} | "
              f"val loss {val['loss']:.4f} acc {val['accuracy']:.3f}", flush=True)
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    metrics, p, y = evaluate(model, validation_loader, device)
    return history, best_epoch, metrics, p, y


def refit(model, loader, device, cfg, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    history = []
    for epoch in range(1, epochs+1):
        row = train_epoch(model, loader, optimizer, device)
        row["epoch"] = epoch
        history.append(row)
        print(f"  Final refit | epoch {epoch}/{epochs} | "
              f"loss {row['train_loss']:.4f} acc {row['train_accuracy']:.3f}", flush=True)
    return history

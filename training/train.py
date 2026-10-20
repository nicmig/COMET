"""Session-fold cross-validation training for the per-microphone LeakCNN.

Run with:
    python -m training.train --config configs/baseline_mic1.yaml
    python -m training.train --config configs/baseline_mic1.yaml --folds 3
    python -m training.train --config configs/baseline_mic1.yaml --set training.epochs=2

Folds: validate on session k, train on the other two sessions (whole-session
splits, no leakage). Artifacts per fold go to
``results/runs/<run_name>/fold<k>/``: best.pt (weights + config + norm stats),
history.csv, metrics.json, val_predictions.npz. A cv_summary.json aggregates
the folds.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from models.cnn import build_model
from monitoring.knn import extract_activations
from preprocessing.dataset import (
    SpectrogramPatchDataset,
    class_names,
    label_to_idx,
    select_rows,
)

log = logging.getLogger("train")


def load_config(path: Path, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
        log.info("override: %s = %r", key, value)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020) on L2-normalized
    projections z with binary labels y. Anchors with no positive in the batch
    are excluded."""
    n = len(z)
    sim = (z @ z.T) / temperature
    not_self = ~torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # stability
    log_prob = sim - torch.log((sim.exp() * not_self).sum(dim=1, keepdim=True))
    pos = (y[:, None] == y[None, :]) & not_self
    n_pos = pos.sum(dim=1)
    mean_log_prob_pos = (pos * log_prob).sum(dim=1) / n_pos.clamp(min=1)
    return -mean_log_prob_pos[n_pos > 0].mean()


def run_epoch(model, loader, criterion, device, optimizer=None,
              supcon: dict | None = None) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if training and supcon:
                # second forward through the (train-mode) encoder + projection
                # head: independent Gaussian-noise/dropout realization, i.e.
                # the standard two-view SupCon setup
                z = model.project(x)
                loss = loss + supcon["lambda"] * supcon_loss(z, y, supcon["temperature"])
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += len(y)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=-1).cpu().numpy())
    return np.concatenate(probs)


def make_loader(ds, cfg, shuffle: bool) -> DataLoader:
    # The dataset holds all spectrograms in memory, so worker processes would
    # only duplicate RAM; num_workers is forced to 0.
    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(cfg["training"]["device"] == "cuda"),
    )


def train_fold(cfg, index: pd.DataFrame, val_session: int, device: torch.device) -> dict:
    classes = class_names(cfg)
    train_sessions = [s for s in (1, 2, 3) if s != val_session]
    log.info("fold val_session=%d  train_sessions=%s", val_session, train_sessions)

    train_rows = select_rows(index, cfg, train_sessions)
    val_rows = select_rows(index, cfg, [val_session])
    log.info("train files=%d  val files=%d", len(train_rows), len(val_rows))

    train_ds = SpectrogramPatchDataset(train_rows, cfg, desc=f"fold{val_session} train audio")
    mean, std = train_ds.compute_norm_stats()
    train_ds.set_norm_stats(mean, std)
    aug = cfg.get("augmentation") or {}
    if aug.get("enabled"):
        train_ds.set_augmentation(aug)
        log.info("augmentation: %s", aug)
    val_ds = SpectrogramPatchDataset(val_rows, cfg, desc=f"fold{val_session} val audio")
    val_ds.set_norm_stats(mean, std)
    log.info("train patches=%d  val patches=%d", len(train_ds), len(val_ds))

    out_dir = Path(cfg["training"]["out_dir"]) / cfg["run_name"] / f"fold{val_session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = Path(cfg["preprocessing"]["norm_stats_dir"]) / cfg["run_name"]
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.savez(stats_dir / f"fold{val_session}.npz", mean=mean, std=std)

    model = build_model(cfg).to(device)
    weight = None
    if cfg["training"].get("class_weights"):
        y_train = np.array([label_to_idx(l, cfg) for l in train_rows["label"]])
        counts = np.bincount(y_train, minlength=len(classes)).astype(float)
        weight = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
        weight = (weight / weight.sum() * len(classes)).to(device)
        log.info("class weights: %s (counts %s)", weight.tolist(), counts.tolist())
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    best_loss, best_epoch, patience_left = float("inf"), 0, cfg["training"]["early_stopping_patience"]
    supcon = cfg.get("supcon") or {}
    supcon = supcon if supcon.get("enabled") else None
    if supcon:
        log.info("supcon: lambda=%.3f temperature=%.3f", supcon["lambda"],
                 supcon["temperature"])
    history = []
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer,
                                    supcon)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": vl_loss, "val_acc": vl_acc})
        log.info("epoch %2d  train loss %.4f acc %.4f | val loss %.4f acc %.4f",
                 epoch, tr_loss, tr_acc, vl_loss, vl_acc)
        if vl_loss < best_loss:
            best_loss, best_epoch = vl_loss, epoch
            patience_left = cfg["training"]["early_stopping_patience"]
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "class_names": classes,
                "val_session": val_session,
                "norm_mean": mean,
                "norm_std": std,
                "epoch": epoch,
                "val_loss": vl_loss,
                "val_acc": vl_acc,
            }, out_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Final evaluation with the best checkpoint.
    ckpt = torch.load(out_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    train_ds.set_augmentation(None)  # activations/predictions on clean patches
    probs = predict(model, val_loader, device)
    y_true = np.array([label_to_idx(val_ds.rows.at[val_ds.row_positions[s], "label"], cfg)
                       for s, _ in val_ds.items], dtype=int)
    y_pred = probs.argmax(1)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    per_class_acc = {
        c: (float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None)
        for i, c in enumerate(classes)
    }
    f1s = []
    for i in range(len(classes)):
        tp = confusion[i, i]
        fp, fn = confusion[:, i].sum() - tp, confusion[i, :].sum() - tp
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    val_f1_macro = float(np.mean(f1s))
    paths = np.array([val_ds.rows.at[val_ds.row_positions[s], "path"]
                      for s, _ in val_ds.items])
    np.savez(out_dir / "val_predictions.npz", probs=probs, y_true=y_true,
             y_pred=y_pred, paths=paths)

    # Activation taps (GAP conv maps + dense) for train/val — the L3 monitor's
    # reference and calibration inputs; saved so the monitor needs no recompute.
    for split, ds in (("train", train_ds), ("val", val_ds)):
        acts, split_probs = extract_activations(model, ds, device)
        split_paths = np.array([ds.rows.at[ds.row_positions[s], "path"]
                                for s, _ in ds.items])
        np.savez(out_dir / f"activations_{split}.npz", probs=split_probs,
                 paths=split_paths,
                 **{f"act_{layer}": acts[layer] for layer in acts})
        log.info("saved %s activations: %d patches", split, len(split_paths))

    metrics = {
        "val_session": val_session,
        "train_sessions": train_sessions,
        "best_epoch": best_epoch,
        "val_loss": best_loss,
        "val_acc": float((y_pred == y_true).mean()),
        "val_f1_macro": val_f1_macro,
        "per_class_acc": per_class_acc,
        "confusion_matrix": confusion.tolist(),
        "class_names": classes,
        "n_train_files": len(train_rows),
        "n_val_files": len(val_rows),
        "n_train_patches": len(train_ds),
        "n_val_patches": len(val_ds),
        "train_class_files": train_rows["label"].value_counts().to_dict(),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("fold %d done: val_acc=%.4f (best epoch %d)", val_session, metrics["val_acc"], best_epoch)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3], help="validation sessions to run")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="a.b.c=value")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    logging.basicConfig(level=getattr(logging, cfg["logging"]["level"], logging.INFO),
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    seed_everything(cfg["seed"])

    device_name = cfg["training"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        log.warning("cuda requested but unavailable; falling back to cpu")
        device_name = "cpu"
    device = torch.device(device_name)
    log.info("device=%s  run=%s  folds=%s", device, cfg["run_name"], args.folds)

    index = pd.read_csv(cfg["data"]["index_csv"])
    log.info("index: %d files", len(index))

    summaries = []
    for fold in args.folds:
        summaries.append(train_fold(cfg, index, fold, device))

    summary = {
        "run_name": cfg["run_name"],
        "folds": [{"val_session": m["val_session"], "val_acc": m["val_acc"],
                   "val_f1_macro": m["val_f1_macro"], "val_loss": m["val_loss"],
                   "best_epoch": m["best_epoch"]}
                  for m in summaries],
        "val_acc_mean": float(np.mean([m["val_acc"] for m in summaries])),
        "val_acc_std": float(np.std([m["val_acc"] for m in summaries])),
        "val_f1_macro_mean": float(np.mean([m["val_f1_macro"] for m in summaries])),
        "val_f1_macro_std": float(np.std([m["val_f1_macro"] for m in summaries])),
    }
    out_dir = Path(cfg["training"]["out_dir"]) / cfg["run_name"]
    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("CV summary: val_acc %.4f ± %.4f -> %s",
             summary["val_acc_mean"], summary["val_acc_std"], out_dir / "cv_summary.json")


if __name__ == "__main__":
    main()

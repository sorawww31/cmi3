"""
Training utilities for CMI Behavior Detection competition.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from schedulefree import RAdamScheduleFree
from timm.utils.model_ema import ModelEmaV3
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .utils import compute_hierarchical_f1


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int = 100,
) -> torch.optim.Optimizer:
    """Optimizer factory function."""
    if optimizer_name == "adam":
        return Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        return AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "radam_schedule_free":
        return RAdamScheduleFree(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> Optional[Any]:
    """Get warmup + cosine annealing scheduler.

    Note: Returns None for RAdamScheduleFree as it handles scheduling internally.
    """
    if isinstance(optimizer, RAdamScheduleFree):
        # RAdamScheduleFree handles scheduling internally
        return None

    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
    )

    # Combine: warmup then cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    return scheduler


def is_schedule_free_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    """Check if optimizer is a schedule-free optimizer."""
    return isinstance(optimizer, RAdamScheduleFree)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[Any] = None,
    ema: Optional[ModelEmaV3] = None,
) -> dict[str, float]:
    """1エポック分の訓練を実行"""
    model.train()

    # RAdamScheduleFree requires optimizer.train() call
    if is_schedule_free_optimizer(optimizer):
        optimizer.train()  # type: ignore

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update EMA after each step
        if ema is not None:
            ema.update(model)

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * data.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader.dataset)  # type: ignore
    f1_score = compute_hierarchical_f1(all_labels, all_preds)

    return {"loss": avg_loss, "f1": f1_score}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, float]:
    """検証を実行"""
    model.eval()

    # RAdamScheduleFree requires optimizer.eval() call for correct param values
    if optimizer is not None and is_schedule_free_optimizer(optimizer):
        optimizer.eval()  # type: ignore

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for batch in pbar:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)

        logits = model(data)
        loss = criterion(logits, labels)

        total_loss += loss.item() * data.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)  # type: ignore
    f1_score = compute_hierarchical_f1(all_labels, all_preds)

    return {"loss": avg_loss, "f1": f1_score}


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Any,
    fold: int,
    output_dir: Path,
    logger: logging.Logger,
    device: torch.device,
) -> dict[str, Any]:
    """1 foldの訓練を実行"""

    criterion = nn.CrossEntropyLoss()

    # Get optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_name=cfg.exp.optimizer_name,
        learning_rate=cfg.exp.learning_rate,
        weight_decay=cfg.exp.weight_decay,
        warmup_steps=cfg.exp.warmup_steps,
    )

    # Get scheduler (None for RAdamScheduleFree)
    total_steps = len(train_loader) * cfg.exp.epochs
    scheduler = get_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=cfg.exp.warmup_steps,
    )

    # Initialize EMA
    ema: Optional[ModelEmaV3] = None
    if cfg.exp.use_ema:
        ema = ModelEmaV3(model, decay=cfg.exp.ema_decay)
        logger.info(f"Using EMA with decay={cfg.exp.ema_decay}")

    logger.info(f"Optimizer: {cfg.exp.optimizer_name}")
    if scheduler is not None:
        logger.info(f"Scheduler: warmup({cfg.exp.warmup_steps}) + cosine")
    else:
        logger.info("Scheduler: internal (RAdamScheduleFree)")

    early_stopping = EarlyStopping(
        patience=cfg.exp.patience,
        mode="max",
    )

    best_score = 0.0
    best_epoch = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_f1": [],
        "val_loss": [],
        "val_f1": [],
    }

    for epoch in range(cfg.exp.epochs):
        # Training
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler, ema
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])

        # Validation - use EMA model if available
        if ema is not None:
            val_metrics = validate(ema.module, val_loader, criterion, device, optimizer)
        else:
            val_metrics = validate(model, val_loader, criterion, device, optimizer)

        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])

        logger.info(
            f"Fold {fold} - Epoch {epoch + 1}/{cfg.exp.epochs} Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f} Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}"
        )

        # Get current learning rate for logging
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        # Log to wandb
        wandb.log(
            {
                f"fold_{fold}/epoch": epoch + 1,
                f"fold_{fold}/train_loss": train_metrics["loss"],
                f"fold_{fold}/train_f1": train_metrics["f1"],
                f"fold_{fold}/val_loss": val_metrics["loss"],
                f"fold_{fold}/val_f1": val_metrics["f1"],
                f"fold_{fold}/learning_rate": current_lr,
            }
        )

        # Check for best model
        is_best = early_stopping(val_metrics["f1"])
        if is_best:
            best_score = val_metrics["f1"]
            best_epoch = epoch + 1
            # Save best model (use EMA weights if available)
            model_path = output_dir / f"model_fold{fold}_best.pt"
            if ema is not None:
                torch.save(ema.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            logger.info(f"  Saved best model to {model_path}")

        # Early stopping
        if early_stopping.early_stop:
            logger.info(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    model_path = output_dir / f"model_fold{fold}_best.pt"
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return {
        "best_score": best_score,
        "best_epoch": best_epoch,
        "history": history,
    }


@torch.no_grad()
def inference(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """推論を実行"""
    model.eval()
    all_preds = []
    all_probs = []

    for batch in tqdm(data_loader, desc="Inference"):
        data = batch["data"].to(device)
        logits = model(data)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.array(all_preds), np.vstack(all_probs)

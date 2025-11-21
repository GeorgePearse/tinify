# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
# BSD 3-Clause Clear License (see LICENSE file)

"""Unified training module for CompressAI models."""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from tinify.datasets import ImageFolder, VideoFolder
from tinify.losses import RateDistortionLoss
from tinify.optimizers import net_aux_optimizer
from tinify.registry import MODELS
from tinify.zoo import image_models, video_models

from .config import Config


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def get_model(config: Config):
    """Get model instance from config."""
    model_name = config.model.name
    quality = config.model.quality
    kwargs = config.model.kwargs

    # Try zoo first (has pretrained weights)
    if config.domain == "image":
        if model_name in image_models:
            return image_models[model_name](quality=quality, pretrained=config.model.pretrained, **kwargs)
    elif config.domain == "video":
        if model_name in video_models:
            return video_models[model_name](quality=quality, pretrained=config.model.pretrained, **kwargs)

    # Fall back to registry
    if model_name in MODELS:
        return MODELS[model_name](**kwargs)

    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")


def get_dataset(config: Config, split: str):
    """Get dataset instance from config."""
    patch_size = tuple(config.dataset.patch_size)

    if split == "train":
        transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
        ])

    if config.domain == "image":
        return ImageFolder(
            config.dataset.path,
            split=split,
            transform=transform,
        )
    elif config.domain == "video":
        if split == "train":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(patch_size),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(patch_size),
            ])
        return VideoFolder(
            config.dataset.path,
            rnd_interval=(split == "train"),
            rnd_temp_order=(split == "train"),
            split=split,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported domain for dataset: {config.domain}")


def configure_optimizers(net, config: Config):
    """Configure optimizers from config."""
    conf = {
        "net": {
            "type": config.optimizer_net.type,
            "lr": config.optimizer_net.lr,
        },
        "aux": {
            "type": config.optimizer_aux.type,
            "lr": config.optimizer_aux.lr,
        },
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch: int,
    config: Config,
):
    """Train for one epoch."""
    model.train()
    device = next(model.parameters()).device
    domain = config.domain

    for i, d in enumerate(train_dataloader):
        if domain == "video":
            d = [frames.to(device) for frames in d]
        else:
            d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()

        if config.training.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_max_norm)

        optimizer.step()

        aux_loss = model.aux_loss()
        if isinstance(aux_loss, list):
            aux_loss = sum(aux_loss)
        aux_loss.backward()
        aux_optimizer.step()

        if i % config.training.log_interval == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d) if domain != 'video' else i}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion.get("mse_loss", out_criterion.get("ms_ssim_loss", 0)):.5f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch: int, test_dataloader, model, criterion, config: Config):
    """Evaluate for one epoch."""
    model.eval()
    device = next(model.parameters()).device
    domain = config.domain

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            if domain == "video":
                d = [frames.to(device) for frames in d]
            else:
                d = d.to(device)

            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux = model.aux_loss()
            if isinstance(aux, list):
                aux = sum(aux)

            aux_loss.update(aux)
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion.get("mse_loss", out_criterion.get("ms_ssim_loss", 0)))

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.5f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state: Dict[str, Any], is_best: bool, save_dir: str, filename: str = "checkpoint.pth.tar"):
    """Save checkpoint to disk."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / filename
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, save_path / "checkpoint_best_loss.pth.tar")


def train(config: Config):
    """Main training function."""
    # Set seed for reproducibility
    if config.training.seed is not None:
        torch.manual_seed(config.training.seed)
        random.seed(config.training.seed)

    # Setup device
    device = "cuda" if config.training.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create datasets
    print(f"Loading dataset from: {config.dataset.path}")
    train_dataset = get_dataset(config, config.dataset.split_train)
    test_dataset = get_dataset(config, config.dataset.split_test)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.training.test_batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # Create model
    print(f"Creating model: {config.model.name} (quality={config.model.quality})")
    net = get_model(config)
    net = net.to(device)

    # Multi-GPU support
    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        net = CustomDataParallel(net)

    # Setup optimizers and scheduler
    optimizer, aux_optimizer = configure_optimizers(net, config)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.scheduler.mode,
        factor=config.scheduler.factor,
        patience=config.scheduler.patience,
        min_lr=config.scheduler.min_lr,
    )

    # Setup loss
    criterion = RateDistortionLoss(lmbda=config.training.lmbda, metric=config.training.metric)

    # Load checkpoint if resuming
    last_epoch = 0
    best_loss = float("inf")

    if config.training.checkpoint:
        print(f"Loading checkpoint: {config.training.checkpoint}")
        checkpoint = torch.load(config.training.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # Training loop
    print(f"\nStarting training for {config.training.epochs} epochs...")
    print(f"Lambda: {config.training.lmbda}, Metric: {config.training.metric}")
    print("-" * 80)

    for epoch in range(last_epoch, config.training.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            config,
        )

        loss = test_epoch(epoch, test_dataloader, net, criterion, config)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if config.training.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "config": config.to_dict(),
                },
                is_best,
                config.training.save_dir,
            )

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {config.training.save_dir}")


def list_models(domain: Optional[str] = None):
    """List available models."""
    print("\nAvailable Models")
    print("=" * 60)

    if domain is None or domain == "image":
        print("\nImage Compression Models:")
        print("-" * 40)
        for name in sorted(image_models.keys()):
            print(f"  {name}")

    if domain is None or domain == "video":
        print("\nVideo Compression Models:")
        print("-" * 40)
        for name in sorted(video_models.keys()):
            print(f"  {name}")

    if domain is None or domain == "pointcloud":
        print("\nPoint Cloud Compression Models:")
        print("-" * 40)
        pcc_models = [k for k in MODELS.keys() if "pcc" in k.lower()]
        for name in sorted(pcc_models):
            print(f"  {name}")

    print("\nAll Registered Models:")
    print("-" * 40)
    for name in sorted(MODELS.keys()):
        print(f"  {name}")

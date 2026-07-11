import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ml.src.dataset import (
    SyntheticEnhancementDataset,
    find_image_files,
    split_image_files,
)
from ml.src.image_ops import apply_correction
from ml.src.model import TinyEnhancementModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIRECTORY = (
    REPOSITORY_ROOT / "ml" / "data" / "source"
)

SPLIT_DIRECTORY = (
    REPOSITORY_ROOT / "ml" / "data" / "splits"
)

MODEL_DIRECTORY = (
    REPOSITORY_ROOT / "ml" / "artifacts" / "models"
)

METRICS_DIRECTORY = (
    REPOSITORY_ROOT / "ml" / "artifacts" / "metrics"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_loss(
    model: nn.Module,
    distorted_images: Tensor,
    target_parameters: Tensor,
    clean_images: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    predicted_parameters = model(distorted_images)

    parameter_loss = functional.smooth_l1_loss(
        predicted_parameters,
        target_parameters,
        beta=0.05,
    )

    corrected_images = apply_correction(
        distorted_images,
        predicted_parameters,
    )

    reconstruction_loss = functional.l1_loss(
        corrected_images,
        clean_images,
    )

    total_loss = parameter_loss + (
        0.5 * reconstruction_loss
    )

    parameter_mae = functional.l1_loss(
        predicted_parameters,
        target_parameters,
    )

    return total_loss, parameter_mae, reconstruction_loss


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
) -> dict[str, float]:
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "parameter_mae": 0.0,
        "reconstruction_mae": 0.0,
        "samples": 0,
    }

    context = (
        torch.enable_grad()
        if is_training
        else torch.no_grad()
    )

    with context:
        for (
            distorted_images,
            target_parameters,
            clean_images,
        ) in data_loader:
            distorted_images = distorted_images.to(device)
            target_parameters = target_parameters.to(device)
            clean_images = clean_images.to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            (
                loss,
                parameter_mae,
                reconstruction_mae,
            ) = calculate_loss(
                model,
                distorted_images,
                target_parameters,
                clean_images,
            )

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = distorted_images.shape[0]

            totals["loss"] += (
                loss.item() * batch_size
            )

            totals["parameter_mae"] += (
                parameter_mae.item() * batch_size
            )

            totals["reconstruction_mae"] += (
                reconstruction_mae.item() * batch_size
            )

            totals["samples"] += batch_size

    sample_count = max(totals["samples"], 1)

    return {
        "loss": totals["loss"] / sample_count,
        "parameter_mae":
            totals["parameter_mae"] / sample_count,
        "reconstruction_mae":
            totals["reconstruction_mae"] / sample_count,
    }


def relative_paths(
    paths: list[Path],
) -> list[str]:
    return [
        str(path.relative_to(SOURCE_DIRECTORY))
        for path in paths
    ]


def save_splits(
    train_files: list[Path],
    validation_files: list[Path],
    test_files: list[Path],
    seed: int,
) -> None:
    SPLIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "seed": seed,
        "train": relative_paths(train_files),
        "validation": relative_paths(
            validation_files
        ),
        "test": relative_paths(test_files),
    }

    output_path = (
        SPLIT_DIRECTORY / "splits.json"
    )

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    validation_metrics: dict[str, float],
    output_path: Path,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_metrics":
                validation_metrics,
            "input_size": 96,
        },
        output_path,
    )


def main() -> None:
    arguments = parse_arguments()
    set_seed(arguments.seed)

    image_files = find_image_files(
        SOURCE_DIRECTORY
    )

    if len(image_files) < 12:
        raise RuntimeError(
            "Добавь минимум 12 JPG или PNG файлов "
            "в ml/data/source."
        )

    (
        train_files,
        validation_files,
        test_files,
    ) = split_image_files(
        image_files,
        seed=arguments.seed,
    )

    save_splits(
        train_files,
        validation_files,
        test_files,
        arguments.seed,
    )

    train_dataset = SyntheticEnhancementDataset(
        train_files,
        training=True,
        seed=arguments.seed,
    )

    validation_dataset = (
        SyntheticEnhancementDataset(
            validation_files,
            training=False,
            seed=arguments.seed + 10_000,
        )
    )

    test_dataset = SyntheticEnhancementDataset(
        test_files,
        training=False,
        seed=arguments.seed + 20_000,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = TinyEnhancementModel().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=0.0001,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=arguments.epochs,
        )
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        MODEL_DIRECTORY / "best_model.pt"
    )

    best_validation_loss = float("inf")
    history: list[dict[str, Any]] = []

    print(f"Device: {device}")
    print(f"Train images: {len(train_files)}")
    print(
        f"Validation images: "
        f"{len(validation_files)}"
    )
    print(f"Test images: {len(test_files)}")

    for epoch in range(1, arguments.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
        )

        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            optimizer=None,
        )

        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "learning_rate":
                optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }

        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:02d}/{arguments.epochs} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss="
            f"{validation_metrics['loss']:.5f} "
            f"val_param_mae="
            f"{validation_metrics['parameter_mae']:.5f}"
        )

        if (
            validation_metrics["loss"]
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_metrics["loss"]
            )

            save_checkpoint(
                model,
                epoch,
                validation_metrics,
                checkpoint_path,
            )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
    )

    metrics_payload = {
        "configuration": {
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate":
                arguments.learning_rate,
            "seed": arguments.seed,
            "device": str(device),
        },
        "dataset": {
            "total": len(image_files),
            "train": len(train_files),
            "validation": len(
                validation_files
            ),
            "test": len(test_files),
        },
        "best_epoch": checkpoint["epoch"],
        "best_validation":
            checkpoint["validation_metrics"],
        "test": test_metrics,
        "history": history,
    }

    metrics_path = (
        METRICS_DIRECTORY /
        "training_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Best model: {checkpoint_path}")
    print(
        f"Test loss: "
        f"{test_metrics['loss']:.5f}"
    )
    print(
        f"Test parameter MAE: "
        f"{test_metrics['parameter_mae']:.5f}"
    )


if __name__ == "__main__":
    main()

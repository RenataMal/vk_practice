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

from ml.src.image_ops import apply_correction
from ml.src.lol_balanced_synthetic_dataset import (
    BalancedSyntheticDataset,
    PARAMETER_MAX,
    PARAMETER_MIN,
    find_images,
)
from ml.src.model import TinyEnhancementModel


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[2]

LOL_DIRECTORY = (
    REPOSITORY_ROOT
    / "ml"
    / "data"
    / "lol"
)

MODEL_DIRECTORY = (
    REPOSITORY_ROOT
    / "ml"
    / "artifacts"
    / "models"
)

METRICS_DIRECTORY = (
    REPOSITORY_ROOT
    / "ml"
    / "artifacts"
    / "metrics"
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
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=6,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_model(
    model: TinyEnhancementModel,
) -> None:
    with torch.no_grad():
        model.parameter_min.copy_(
            PARAMETER_MIN.to(
                model.parameter_min.device
            )
        )

        model.parameter_max.copy_(
            PARAMETER_MAX.to(
                model.parameter_max.device
            )
        )

        final_linear = model.regressor[3]

        if not isinstance(
            final_linear,
            nn.Linear,
        ):
            raise TypeError(
                "В regressor[3] ожидался Linear."
            )

        neutral = torch.ones(
            3,
            dtype=torch.float32,
            device=model.parameter_min.device,
        )

        ratio = (
            (neutral - model.parameter_min)
            / (
                model.parameter_max
                - model.parameter_min
            )
        ).clamp(
            0.001,
            0.999,
        )

        neutral_bias = torch.log(
            ratio / (1.0 - ratio)
        )

        nn.init.normal_(
            final_linear.weight,
            mean=0.0,
            std=0.01,
        )

        final_linear.bias.copy_(
            neutral_bias
        )


def calculate_loss(
    model: TinyEnhancementModel,
    input_images: Tensor,
    target_parameters: Tensor,
    target_images: Tensor,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    predicted_parameters = model(
        input_images
    )

    parameter_scale = (
        model.parameter_max
        - model.parameter_min
    )

    normalized_prediction = (
        predicted_parameters
        - model.parameter_min
    ) / parameter_scale

    normalized_target = (
        target_parameters
        - model.parameter_min
    ) / parameter_scale

    parameter_loss = (
        functional.smooth_l1_loss(
            normalized_prediction,
            normalized_target,
            beta=0.05,
        )
    )

    corrected_images = apply_correction(
        input_images,
        predicted_parameters,
    )

    pixel_loss = functional.l1_loss(
        corrected_images,
        target_images,
    )

    mean_loss = functional.l1_loss(
        corrected_images.mean(
            dim=(2, 3)
        ),
        target_images.mean(
            dim=(2, 3)
        ),
    )

    predicted_std = (
        predicted_parameters.std(
            dim=0,
            unbiased=False,
        )
    )

    target_std = (
        target_parameters.std(
            dim=0,
            unbiased=False,
        )
    )

    variance_loss = (
        functional.l1_loss(
            predicted_std,
            target_std,
        )
    )

    total_loss = (
        parameter_loss
        + 0.25 * pixel_loss
        + 0.10 * mean_loss
        + 0.20 * variance_loss
    )

    return (
        total_loss,
        parameter_loss,
        pixel_loss,
        variance_loss,
        predicted_parameters,
    )


def initialize_statistics() -> dict[
    str,
    Tensor | None,
]:
    return {
        "sum": torch.zeros(3),
        "sum_square": torch.zeros(3),
        "minimum": None,
        "maximum": None,
    }


def update_statistics(
    statistics: dict[
        str,
        Tensor | None,
    ],
    values: Tensor,
) -> None:
    values = values.detach().cpu()

    statistics["sum"] = (
        statistics["sum"]
        + values.sum(dim=0)
    )

    statistics["sum_square"] = (
        statistics["sum_square"]
        + values.square().sum(dim=0)
    )

    batch_minimum = (
        values.min(dim=0).values
    )

    batch_maximum = (
        values.max(dim=0).values
    )

    current_minimum = (
        statistics["minimum"]
    )

    current_maximum = (
        statistics["maximum"]
    )

    statistics["minimum"] = (
        batch_minimum
        if current_minimum is None
        else torch.minimum(
            current_minimum,
            batch_minimum,
        )
    )

    statistics["maximum"] = (
        batch_maximum
        if current_maximum is None
        else torch.maximum(
            current_maximum,
            batch_maximum,
        )
    )


def finalize_statistics(
    statistics: dict[
        str,
        Tensor | None,
    ],
    count: int,
) -> dict[str, list[float]]:
    divisor = max(count, 1)

    values_sum = statistics["sum"]
    values_sum_square = (
        statistics["sum_square"]
    )

    if (
        values_sum is None
        or values_sum_square is None
    ):
        raise RuntimeError(
            "Статистика параметров пуста."
        )

    mean = values_sum / divisor

    variance = (
        values_sum_square / divisor
        - mean.square()
    ).clamp_min(0.0)

    minimum = statistics["minimum"]
    maximum = statistics["maximum"]

    if minimum is None or maximum is None:
        minimum = torch.zeros(3)
        maximum = torch.zeros(3)

    return {
        "mean": mean.tolist(),
        "std": (
            variance.sqrt().tolist()
        ),
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "range": (
            (maximum - minimum).tolist()
        ),
    }


def run_epoch(
    model: TinyEnhancementModel,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
) -> dict[str, Any]:
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "parameter_loss": 0.0,
        "pixel_loss": 0.0,
        "variance_loss": 0.0,
        "parameter_mae": 0.0,
        "samples": 0,
    }

    predicted_statistics = (
        initialize_statistics()
    )

    target_statistics = (
        initialize_statistics()
    )

    context = (
        torch.enable_grad()
        if is_training
        else torch.no_grad()
    )

    with context:
        for (
            input_images,
            target_parameters,
            target_images,
        ) in data_loader:
            input_images = input_images.to(
                device
            )

            target_parameters = (
                target_parameters.to(device)
            )

            target_images = (
                target_images.to(device)
            )

            if optimizer is not None:
                optimizer.zero_grad(
                    set_to_none=True
                )

            (
                loss,
                parameter_loss,
                pixel_loss,
                variance_loss,
                predicted_parameters,
            ) = calculate_loss(
                model,
                input_images,
                target_parameters,
                target_images,
            )

            if optimizer is not None:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            batch_size = (
                input_images.shape[0]
            )

            parameter_mae = (
                predicted_parameters
                .sub(target_parameters)
                .abs()
                .mean()
            )

            totals["loss"] += (
                loss.item()
                * batch_size
            )

            totals["parameter_loss"] += (
                parameter_loss.item()
                * batch_size
            )

            totals["pixel_loss"] += (
                pixel_loss.item()
                * batch_size
            )

            totals["variance_loss"] += (
                variance_loss.item()
                * batch_size
            )

            totals["parameter_mae"] += (
                parameter_mae.item()
                * batch_size
            )

            totals["samples"] += batch_size

            update_statistics(
                predicted_statistics,
                predicted_parameters,
            )

            update_statistics(
                target_statistics,
                target_parameters,
            )

    sample_count = max(
        totals["samples"],
        1,
    )

    return {
        "loss": (
            totals["loss"]
            / sample_count
        ),
        "parameter_loss": (
            totals["parameter_loss"]
            / sample_count
        ),
        "pixel_mae": (
            totals["pixel_loss"]
            / sample_count
        ),
        "variance_loss": (
            totals["variance_loss"]
            / sample_count
        ),
        "parameter_mae": (
            totals["parameter_mae"]
            / sample_count
        ),
        "samples": totals["samples"],
        "predicted_parameters": (
            finalize_statistics(
                predicted_statistics,
                totals["samples"],
            )
        ),
        "target_parameters": (
            finalize_statistics(
                target_statistics,
                totals["samples"],
            )
        ),
    }


def format_values(
    values: list[float],
) -> str:
    return (
        "["
        + ", ".join(
            f"{value:.4f}"
            for value in values
        )
        + "]"
    )


def main() -> None:
    arguments = parse_arguments()
    set_seed(arguments.seed)

    train_images = find_images(
        LOL_DIRECTORY
        / "our485"
        / "high"
    )

    test_images = find_images(
        LOL_DIRECTORY
        / "eval15"
        / "high"
    )

    shuffled = list(train_images)
    random.Random(
        arguments.seed
    ).shuffle(shuffled)

    validation_size = max(
        1,
        round(
            len(shuffled) * 0.15
        ),
    )

    validation_images = shuffled[
        :validation_size
    ]

    training_images = shuffled[
        validation_size:
    ]

    train_dataset = (
        BalancedSyntheticDataset(
            training_images,
            training=True,
            variants_per_image=8,
            seed=arguments.seed,
        )
    )

    validation_dataset = (
        BalancedSyntheticDataset(
            validation_images,
            training=False,
            variants_per_image=6,
            seed=arguments.seed + 10000,
        )
    )

    test_dataset = (
        BalancedSyntheticDataset(
            test_images,
            training=False,
            variants_per_image=12,
            seed=arguments.seed + 20000,
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = TinyEnhancementModel().to(
        device
    )

    configure_model(model)

    optimizer = AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=0.0001,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=0.00001,
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
        MODEL_DIRECTORY
        / "best_balanced_synthetic_model.pt"
    )

    best_validation_loss = (
        float("inf")
    )

    epochs_without_improvement = 0
    history: list[
        dict[str, Any]
    ] = []

    print(f"Device: {device}")

    print(
        f"Train images: "
        f"{len(training_images)}"
    )

    print(
        f"Train samples per epoch: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(validation_images)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Test images: "
        f"{len(test_images)}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    for epoch in range(
        1,
        arguments.epochs + 1,
    ):
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

        scheduler.step(
            validation_metrics["loss"]
        )

        history.append(
            {
                "epoch": epoch,
                "learning_rate": (
                    optimizer.param_groups[0][
                        "lr"
                    ]
                ),
                "train": train_metrics,
                "validation": (
                    validation_metrics
                ),
            }
        )

        predicted_std = (
            validation_metrics[
                "predicted_parameters"
            ]["std"]
        )

        target_std = (
            validation_metrics[
                "target_parameters"
            ]["std"]
        )

        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{arguments.epochs} "
            f"train_loss="
            f"{train_metrics['loss']:.5f} "
            f"val_loss="
            f"{validation_metrics['loss']:.5f} "
            f"val_param_mae="
            f"{validation_metrics['parameter_mae']:.5f} "
            f"pred_std="
            f"{format_values(predicted_std)} "
            f"target_std="
            f"{format_values(target_std)}"
        )

        if (
            validation_metrics["loss"]
            < best_validation_loss
            - 0.0001
        ):
            best_validation_loss = (
                validation_metrics["loss"]
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "epoch": epoch,
                    "validation_metrics": (
                        validation_metrics
                    ),
                    "input_size": 96,
                    "dataset": (
                        "LOL-high-balanced-synthetic"
                    ),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= arguments.patience
        ):
            print(
                "Early stopping: "
                "validation loss "
                "не улучшался."
            )

            break

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
            "epochs_requested": (
                arguments.epochs
            ),
            "epochs_completed": (
                len(history)
            ),
            "batch_size": (
                arguments.batch_size
            ),
            "learning_rate": (
                arguments.learning_rate
            ),
            "seed": arguments.seed,
            "patience": (
                arguments.patience
            ),
            "device": str(device),
        },
        "dataset": {
            "name": (
                "LOL-high-balanced-synthetic"
            ),
            "train_images": (
                len(training_images)
            ),
            "train_samples_per_epoch": (
                len(train_dataset)
            ),
            "validation_images": (
                len(validation_images)
            ),
            "validation_samples": (
                len(validation_dataset)
            ),
            "test_images": (
                len(test_images)
            ),
            "test_samples": (
                len(test_dataset)
            ),
        },
        "best_epoch": (
            checkpoint["epoch"]
        ),
        "best_validation": (
            checkpoint[
                "validation_metrics"
            ]
        ),
        "test": test_metrics,
        "history": history,
    }

    metrics_path = (
        METRICS_DIRECTORY
        / "balanced_synthetic_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Best model: "
        f"{checkpoint_path}"
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Test loss: "
        f"{test_metrics['loss']:.5f}"
    )

    print(
        f"Test parameter MAE: "
        f"{test_metrics['parameter_mae']:.5f}"
    )

    print(
        "Test predicted std: "
        f"{format_values(
            test_metrics[
                'predicted_parameters'
            ]['std']
        )}"
    )

    print(
        "Test target std: "
        f"{format_values(
            test_metrics[
                'target_parameters'
            ]['std']
        )}"
    )

    print(
        "Test predicted range: "
        f"{format_values(
            test_metrics[
                'predicted_parameters'
            ]['range']
        )}"
    )


if __name__ == "__main__":
    main()

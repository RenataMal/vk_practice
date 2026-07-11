import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.src.model import TinyEnhancementModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    REPOSITORY_ROOT
    / "ml"
    / "artifacts"
    / "models"
    / "best_model.pt"
)

OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "app"
    / "public"
    / "models"
)

CONFIG_PATH = OUTPUT_DIRECTORY / "model-config.json"
WEIGHTS_PATH = OUTPUT_DIRECTORY / "model-weights.bin"

TENSOR_NAMES = [
    "features.0.weight",
    "features.0.bias",
    "features.2.weight",
    "features.2.bias",
    "features.4.weight",
    "features.4.bias",
    "features.6.weight",
    "features.6.bias",
    "regressor.1.weight",
    "regressor.1.bias",
    "regressor.3.weight",
    "regressor.3.bias",
]


def load_checkpoint() -> dict[str, Any]:
    try:
        return torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
        )


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Файл best_model.pt не найден."
        )

    checkpoint = load_checkpoint()

    model = TinyEnhancementModel()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    state_dict = model.state_dict()

    descriptors: dict[str, dict[str, Any]] = {}
    arrays: list[np.ndarray] = []
    offset = 0

    for name in TENSOR_NAMES:
        tensor = (
            state_dict[name]
            .detach()
            .cpu()
            .contiguous()
        )

        array = (
            tensor.numpy()
            .astype("<f4", copy=False)
            .reshape(-1)
        )

        descriptors[name] = {
            "offset": offset,
            "length": int(array.size),
            "shape": list(tensor.shape),
        }

        arrays.append(array)
        offset += int(array.size)

    weights = np.concatenate(arrays)

    config = {
        "version": 1,
        "inputSize": 96,
        "parameterMin": [
            float(value)
            for value in model.parameter_min.tolist()
        ],
        "parameterMax": [
            float(value)
            for value in model.parameter_max.tolist()
        ],
        "tensors": descriptors,
    }

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    WEIGHTS_PATH.write_bytes(
        weights.tobytes(order="C")
    )

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Config: {CONFIG_PATH}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(
        f"Parameters: {weights.size:,}"
    )
    print(
        f"Size: {WEIGHTS_PATH.stat().st_size / 1024:.2f} KB"
    )


if __name__ == "__main__":
    main()

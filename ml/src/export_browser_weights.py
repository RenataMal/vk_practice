import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.src.model import TinyEnhancementModel


DEFAULT_CHECKPOINT_PATH = (
    REPOSITORY_ROOT
    / "ml"
    / "artifacts"
    / "models"
    / "best_balanced_synthetic_model.pt"
)

DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "app"
    / "public"
    / "models"
)

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    parser.add_argument(
        "--verify",
        action="store_true",
    )

    return parser.parse_args()


def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint должен содержать словарь.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("State dict модели не найден.")

    return state_dict


def build_export(
    checkpoint_path: Path,
) -> tuple[str, bytes, int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint не найден: {checkpoint_path}"
        )

    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)

    model = TinyEnhancementModel()
    model.load_state_dict(
        state_dict,
        strict=True,
    )
    model.eval()

    exported_state = model.state_dict()

    descriptors: dict[str, dict[str, Any]] = {}
    arrays: list[np.ndarray] = []
    offset = 0

    for name in TENSOR_NAMES:
        tensor = (
            exported_state[name]
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
    weights_bytes = weights.tobytes(order="C")

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

    config_text = json.dumps(
        config,
        ensure_ascii=False,
        indent=2,
    )

    return config_text, weights_bytes, int(weights.size)


def verify_export(
    config_path: Path,
    weights_path: Path,
    expected_config: str,
    expected_weights: bytes,
) -> None:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Конфигурация не найдена: {config_path}"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Веса не найдены: {weights_path}"
        )

    current_config = config_path.read_text(
        encoding="utf-8",
    )
    current_weights = weights_path.read_bytes()

    config_matches = current_config == expected_config
    weights_match = current_weights == expected_weights

    print(
        "Config:",
        "MATCH" if config_matches else "NO MATCH",
    )
    print(
        "Weights:",
        "MATCH" if weights_match else "NO MATCH",
    )
    print(
        "SHA256:",
        sha256(current_weights).hexdigest(),
    )

    if not config_matches or not weights_match:
        raise SystemExit(1)


def write_export(
    output_directory: Path,
    config_text: str,
    weights_bytes: bytes,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = (
        output_directory
        / "model-config.json"
    )
    weights_path = (
        output_directory
        / "model-weights.bin"
    )

    config_path.write_text(
        config_text,
        encoding="utf-8",
    )
    weights_path.write_bytes(
        weights_bytes,
    )

    print(f"Config: {config_path}")
    print(f"Weights: {weights_path}")


def main() -> None:
    arguments = parse_arguments()

    checkpoint_path = arguments.checkpoint.resolve()
    output_directory = (
        arguments.output_directory.resolve()
    )

    config_text, weights_bytes, parameter_count = (
        build_export(checkpoint_path)
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Parameters: {parameter_count:,}")
    print(
        "Size:",
        f"{len(weights_bytes) / 1024:.2f} KB",
    )

    if arguments.verify:
        verify_export(
            output_directory / "model-config.json",
            output_directory / "model-weights.bin",
            config_text,
            weights_bytes,
        )
        return

    write_export(
        output_directory,
        config_text,
        weights_bytes,
    )


if __name__ == "__main__":
    main()
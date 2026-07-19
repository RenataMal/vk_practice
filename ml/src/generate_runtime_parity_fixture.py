import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.src.model import TinyEnhancementModel


CHECKPOINT_PATH = (
    REPOSITORY_ROOT
    / "ml"
    / "artifacts"
    / "models"
    / "best_balanced_synthetic_model.pt"
)

OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "app"
    / "src"
    / "model"
    / "runtime-parity.fixture.json"
)

INPUT_SIZE = 96


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


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, Any]:
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


def create_input(name: str) -> torch.Tensor:
    channels = torch.arange(
        3,
        dtype=torch.float32,
    ).reshape(3, 1, 1)

    vertical = torch.arange(
        INPUT_SIZE,
        dtype=torch.float32,
    ).reshape(1, INPUT_SIZE, 1)

    horizontal = torch.arange(
        INPUT_SIZE,
        dtype=torch.float32,
    ).reshape(1, 1, INPUT_SIZE)

    if name == "zeros":
        image = torch.zeros(
            3,
            INPUT_SIZE,
            INPUT_SIZE,
            dtype=torch.float32,
        )
    elif name == "ones":
        image = torch.ones(
            3,
            INPUT_SIZE,
            INPUT_SIZE,
            dtype=torch.float32,
        )
    elif name == "gradient":
        image = (
            (
                channels * 13
                + vertical * 3
                + horizontal * 5
            )
            % 256
        ) / 255
    elif name == "checkerboard":
        image = (
            (
                torch.floor(horizontal / 8)
                + torch.floor(vertical / 8)
                + channels
            )
            % 2
        )
    else:
        raise ValueError(
            f"Неизвестный тестовый вход: {name}"
        )

    return image.unsqueeze(0)


def round_like_javascript(
    value: float,
    digits: int = 3,
) -> float:
    multiplier = 10 ** digits

    return (
        math.floor(value * multiplier + 0.5)
        / multiplier
    )


def main() -> None:
    checkpoint = load_checkpoint(
        CHECKPOINT_PATH
    )

    model = TinyEnhancementModel()
    model.load_state_dict(
        extract_state_dict(checkpoint),
        strict=True,
    )
    model.eval()

    cases = []

    with torch.no_grad():
        for name in [
            "zeros",
            "ones",
            "gradient",
            "checkerboard",
        ]:
            prediction = model(
                create_input(name)
            )[0].tolist()

            cases.append(
                {
                    "name": name,
                    "expected": {
                        "brightness": round_like_javascript(
                            float(prediction[0])
                        ),
                        "contrast": round_like_javascript(
                            float(prediction[1])
                        ),
                        "saturation": round_like_javascript(
                            float(prediction[2])
                        ),
                    },
                }
            )

    fixture = {
        "inputSize": INPUT_SIZE,
        "checkpoint": CHECKPOINT_PATH.name,
        "cases": cases,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            fixture,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Fixture: {OUTPUT_PATH}")

    for case in cases:
        print(
            case["name"],
            case["expected"],
        )


if __name__ == "__main__":
    main()
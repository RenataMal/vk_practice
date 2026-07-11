import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ml.src.model import TinyEnhancementModel


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[2]

CHECKPOINT_PATH = (
    REPOSITORY_ROOT
    / "ml"
    / "artifacts"
    / "models"
    / "best_balanced_synthetic_model.pt"
)

DEFAULT_DIRECTORY = (
    REPOSITORY_ROOT
    / "ml"
    / "data"
    / "diagnostic"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_DIRECTORY,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    directory = arguments.directory

    if not directory.is_absolute():
        directory = (
            REPOSITORY_ROOT
            / directory
        )

    image_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"В папке нет изображений: "
            f"{directory}"
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
        )

    model = TinyEnhancementModel()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    preprocessing = transforms.Compose(
        [
            transforms.Resize(
                112,
                antialias=True,
            ),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
        ]
    )

    predictions = []

    with torch.no_grad():
        for image_path in image_paths:
            with Image.open(
                image_path
            ) as image:
                tensor = preprocessing(
                    image.convert("RGB")
                ).unsqueeze(0)

            values = model(tensor)[0]

            prediction = [
                float(value)
                for value in values
            ]

            predictions.append(
                prediction
            )

            print(
                f"{image_path.name}: "
                f"brightness="
                f"{prediction[0]:.4f}, "
                f"contrast="
                f"{prediction[1]:.4f}, "
                f"saturation="
                f"{prediction[2]:.4f}"
            )

    prediction_tensor = torch.tensor(
        predictions,
        dtype=torch.float32,
    )

    print()

    print(
        "Standard deviation:",
        prediction_tensor.std(
            dim=0,
            unbiased=False,
        ).tolist(),
    )

    print(
        "Range:",
        (
            prediction_tensor.max(
                dim=0
            ).values
            - prediction_tensor.min(
                dim=0
            ).values
        ).tolist(),
    )


if __name__ == "__main__":
    main()

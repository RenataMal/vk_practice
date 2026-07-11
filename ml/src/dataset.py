import random
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from ml.src.image_ops import create_distorted_image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

PARAMETER_MIN = torch.tensor(
    [0.80, 0.88, 0.88],
    dtype=torch.float32,
)

PARAMETER_MAX = torch.tensor(
    [1.28, 1.35, 1.30],
    dtype=torch.float32,
)


def find_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    files = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


def split_image_files(
    files: Sequence[Path],
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[list[Path], list[Path], list[Path]]:
    if len(files) < 12:
        raise ValueError(
            "Для обучения нужно минимум 12 изображений."
        )

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    validation_size = max(
        1,
        round(total * validation_fraction),
    )
    test_size = max(
        1,
        round(total * test_fraction),
    )
    train_size = total - validation_size - test_size

    if train_size < 1:
        raise ValueError(
            "Недостаточно изображений для train-разбиения."
        )

    train_files = shuffled[:train_size]

    validation_files = shuffled[
        train_size:train_size + validation_size
    ]

    test_files = shuffled[
        train_size + validation_size:
    ]

    return train_files, validation_files, test_files


class SyntheticEnhancementDataset(Dataset):
    def __init__(
        self,
        image_files: Sequence[Path],
        training: bool,
        input_size: int = 96,
        seed: int = 42,
        identity_probability: float = 0.25,
    ) -> None:
        self.image_files = list(image_files)
        self.training = training
        self.seed = seed
        self.identity_probability = identity_probability

        if training:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        input_size,
                        scale=(0.72, 1.0),
                        ratio=(0.80, 1.25),
                        antialias=True,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            resize_size = round(input_size * 1.17)

            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        resize_size,
                        antialias=True,
                    ),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def create_target_parameters(
        self,
        index: int,
    ) -> Tensor:
        generator = None

        if not self.training:
            generator = torch.Generator()
            generator.manual_seed(self.seed + index)

        identity_random = torch.rand(
            1,
            generator=generator,
        ).item()

        if identity_random < self.identity_probability:
            return torch.ones(3, dtype=torch.float32)

        random_values = torch.rand(
            3,
            generator=generator,
        )

        return PARAMETER_MIN + random_values * (
            PARAMETER_MAX - PARAMETER_MIN
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        image_path = self.image_files[index]

        try:
            with Image.open(image_path) as image:
                clean_image = self.transform(
                    image.convert("RGB")
                )
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(
                f"Не удалось открыть {image_path}"
            ) from error

        target_parameters = (
            self.create_target_parameters(index)
        )

        distorted_image = create_distorted_image(
            clean_image,
            target_parameters,
        )

        return (
            distorted_image,
            target_parameters,
            clean_image,
        )

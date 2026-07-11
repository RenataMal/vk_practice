from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as transform_functional

from ml.src.image_ops import create_distorted_image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

PARAMETER_MIN = torch.tensor(
    [0.65, 0.75, 0.70],
    dtype=torch.float32,
)

PARAMETER_MAX = torch.tensor(
    [1.65, 1.35, 1.40],
    dtype=torch.float32,
)


def find_images(
    directory: Path,
) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Папка не найдена: {directory}"
        )

    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not images:
        raise RuntimeError(
            f"В папке нет изображений: {directory}"
        )

    return images


def random_value(
    generator: torch.Generator | None,
) -> float:
    return float(
        torch.rand(
            (),
            generator=generator,
        ).item()
    )


def sample_away_from_neutral(
    minimum: float,
    maximum: float,
    generator: torch.Generator | None,
) -> float:
    side = random_value(generator)
    position = random_value(generator)

    if side < 0.5:
        return (
            minimum
            + position
            * (0.93 - minimum)
        )

    return (
        1.07
        + position
        * (maximum - 1.07)
    )


def sample_target_parameters(
    generator: torch.Generator | None,
) -> Tensor:
    mode = random_value(generator)

    parameters = torch.ones(
        3,
        dtype=torch.float32,
    )

    if mode < 0.05:
        return parameters

    if mode < 0.30:
        parameters[0] = sample_away_from_neutral(
            float(PARAMETER_MIN[0]),
            float(PARAMETER_MAX[0]),
            generator,
        )

        return parameters

    if mode < 0.55:
        parameters[1] = sample_away_from_neutral(
            float(PARAMETER_MIN[1]),
            float(PARAMETER_MAX[1]),
            generator,
        )

        return parameters

    if mode < 0.80:
        parameters[2] = sample_away_from_neutral(
            float(PARAMETER_MIN[2]),
            float(PARAMETER_MAX[2]),
            generator,
        )

        return parameters

    parameters[0] = sample_away_from_neutral(
        float(PARAMETER_MIN[0]),
        float(PARAMETER_MAX[0]),
        generator,
    )

    parameters[1] = sample_away_from_neutral(
        float(PARAMETER_MIN[1]),
        float(PARAMETER_MAX[1]),
        generator,
    )

    parameters[2] = sample_away_from_neutral(
        float(PARAMETER_MIN[2]),
        float(PARAMETER_MAX[2]),
        generator,
    )

    return parameters


class BalancedSyntheticDataset(Dataset):
    def __init__(
        self,
        image_files: Sequence[Path],
        training: bool,
        input_size: int = 96,
        variants_per_image: int = 8,
        seed: int = 42,
    ) -> None:
        self.image_files = list(
            image_files
        )

        self.training = training
        self.input_size = input_size
        self.seed = seed
        self.variants_per_image = (
            variants_per_image
        )

        if not self.image_files:
            raise ValueError(
                "Список изображений пуст."
            )

        if variants_per_image < 1:
            raise ValueError(
                "variants_per_image должен быть больше нуля."
            )

        self.image_cache: dict[
            Path,
            Image.Image,
        ] = {}

    def __len__(self) -> int:
        return (
            len(self.image_files)
            * self.variants_per_image
        )

    def load_image(
        self,
        image_path: Path,
    ) -> Image.Image:
        cached = self.image_cache.get(
            image_path
        )

        if cached is not None:
            return cached.copy()

        try:
            with Image.open(
                image_path
            ) as image:
                loaded = image.convert(
                    "RGB"
                )

                loaded.thumbnail(
                    (512, 512),
                    Image.Resampling.LANCZOS,
                )

                cached = loaded.copy()
        except (
            OSError,
            UnidentifiedImageError,
        ) as error:
            raise RuntimeError(
                f"Не удалось открыть {image_path}"
            ) from error

        self.image_cache[image_path] = (
            cached
        )

        return cached.copy()

    def transform_image(
        self,
        image: Image.Image,
        generator: torch.Generator | None,
    ) -> Tensor:
        if self.training:
            top, left, height, width = (
                transforms.RandomResizedCrop
                .get_params(
                    image,
                    scale=(0.60, 1.0),
                    ratio=(0.75, 1.33),
                )
            )

            image = (
                transform_functional
                .resized_crop(
                    image,
                    top,
                    left,
                    height,
                    width,
                    [
                        self.input_size,
                        self.input_size,
                    ],
                    antialias=True,
                )
            )

            if random_value(generator) < 0.5:
                image = (
                    transform_functional
                    .hflip(image)
                )
        else:
            image = (
                transform_functional.resize(
                    image,
                    round(
                        self.input_size
                        * 1.17
                    ),
                    antialias=True,
                )
            )

            image = (
                transform_functional
                .center_crop(
                    image,
                    [
                        self.input_size,
                        self.input_size,
                    ],
                )
            )

        return (
            transform_functional
            .to_tensor(image)
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        image_index = (
            index
            % len(self.image_files)
        )

        generator = None

        if not self.training:
            generator = torch.Generator()
            generator.manual_seed(
                self.seed + index
            )

        clean_image = self.load_image(
            self.image_files[
                image_index
            ]
        )

        clean_tensor = self.transform_image(
            clean_image,
            generator,
        )

        target_parameters = (
            sample_target_parameters(
                generator
            )
        )

        distorted_tensor = (
            create_distorted_image(
                clean_tensor,
                target_parameters,
            )
        )

        return (
            distorted_tensor,
            target_parameters,
            clean_tensor,
        )

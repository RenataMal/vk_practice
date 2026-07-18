import torch
from torch import Tensor, nn


class TinyEnhancementModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                48,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                48,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )

        self.register_buffer(
            "parameter_min",
            torch.tensor(
                [0.65, 0.75, 0.70],
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "parameter_max",
            torch.tensor(
                [1.65, 1.35, 1.40],
                dtype=torch.float32,
            ),
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        normalized_parameters = self.regressor(features)

        return self.parameter_min + normalized_parameters * (
            self.parameter_max - self.parameter_min
        )
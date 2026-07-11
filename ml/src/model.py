import torch
from torch import Tensor, nn


class TinyEnhancementModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                8,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                16,
                24,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                24,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
            nn.Sigmoid(),
        )

        self.register_buffer(
            "parameter_min",
            torch.tensor(
                [0.80, 0.88, 0.88],
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "parameter_max",
            torch.tensor(
                [1.28, 1.35, 1.30],
                dtype=torch.float32,
            ),
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        normalized_parameters = self.regressor(features)

        return self.parameter_min + normalized_parameters * (
            self.parameter_max - self.parameter_min
        )

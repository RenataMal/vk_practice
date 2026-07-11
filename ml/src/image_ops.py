import torch
from torch import Tensor


def apply_correction(
    images: Tensor,
    parameters: Tensor,
) -> Tensor:
    squeeze_result = False

    if images.ndim == 3:
        images = images.unsqueeze(0)
        squeeze_result = True

    if parameters.ndim == 1:
        parameters = parameters.unsqueeze(0)

    if images.ndim != 4:
        raise ValueError("Images must have shape [B, C, H, W].")

    if parameters.ndim != 2 or parameters.shape[1] != 3:
        raise ValueError("Parameters must have shape [B, 3].")

    brightness = parameters[:, 0].reshape(-1, 1, 1, 1)
    contrast = parameters[:, 1].reshape(-1, 1, 1, 1)
    saturation = parameters[:, 2].reshape(-1, 1, 1, 1)

    corrected = images * brightness
    corrected = (corrected - 0.5) * contrast + 0.5

    red = corrected[:, 0:1]
    green = corrected[:, 1:2]
    blue = corrected[:, 2:3]

    grayscale = (
        0.2126 * red +
        0.7152 * green +
        0.0722 * blue
    )

    corrected = grayscale + saturation * (
        corrected - grayscale
    )

    corrected = corrected.clamp(0.0, 1.0)

    if squeeze_result:
        return corrected.squeeze(0)

    return corrected


def create_distorted_image(
    clean_image: Tensor,
    target_parameters: Tensor,
) -> Tensor:
    if clean_image.ndim != 3:
        raise ValueError(
            "Clean image must have shape [C, H, W]."
        )

    if target_parameters.shape != (3,):
        raise ValueError(
            "Target parameters must have shape [3]."
        )

    brightness = target_parameters[0]
    contrast = target_parameters[1]
    saturation = target_parameters[2]

    red = clean_image[0:1]
    green = clean_image[1:2]
    blue = clean_image[2:3]

    grayscale = (
        0.2126 * red +
        0.7152 * green +
        0.0722 * blue
    )

    distorted = grayscale + (
        1.0 / saturation
    ) * (clean_image - grayscale)

    distorted = (
        distorted - 0.5
    ) * (1.0 / contrast) + 0.5

    distorted = distorted * (1.0 / brightness)

    return distorted.clamp(0.0, 1.0)

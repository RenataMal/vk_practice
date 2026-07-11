import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
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

ARTIFACT_ONNX_PATH = (
    REPOSITORY_ROOT
    / "ml"
    / "artifacts"
    / "models"
    / "enhancer.onnx"
)

BROWSER_ONNX_PATH = (
    REPOSITORY_ROOT
    / "app"
    / "public"
    / "models"
    / "enhancer.onnx"
)


def load_checkpoint() -> dict:
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
            "Сначала запусти обучение модели."
        )

    checkpoint = load_checkpoint()

    model = TinyEnhancementModel()
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    dummy_input = torch.rand(
        1,
        3,
        96,
        96,
        dtype=torch.float32,
    )

    ARTIFACT_ONNX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    BROWSER_ONNX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.onnx.export(
        model,
        (dummy_input,),
        str(ARTIFACT_ONNX_PATH),
        input_names=["image"],
        output_names=["parameters"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )

    onnx_model = onnx.load(
        ARTIFACT_ONNX_PATH
    )

    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(ARTIFACT_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    onnx_output = session.run(
        ["parameters"],
        {
            "image": dummy_input.numpy(),
        },
    )[0]

    with torch.no_grad():
        torch_output = model(
            dummy_input
        ).numpy()

    maximum_difference = float(
        np.max(
            np.abs(
                onnx_output - torch_output
            )
        )
    )

    shutil.copy2(
        ARTIFACT_ONNX_PATH,
        BROWSER_ONNX_PATH,
    )

    model_size_kb = (
        BROWSER_ONNX_PATH.stat().st_size
        / 1024
    )

    print(
        f"ONNX model: {BROWSER_ONNX_PATH}"
    )
    print(
        f"Size: {model_size_kb:.2f} KB"
    )
    print(
        f"Maximum difference: "
        f"{maximum_difference:.8f}"
    )


if __name__ == "__main__":
    main()

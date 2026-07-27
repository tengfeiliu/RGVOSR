"""Compatibility helpers for third-party libraries using removed torchvision APIs."""

import importlib
import sys
import types


LEGACY_FUNCTIONAL_TENSOR_MODULE = "torchvision.transforms.functional_tensor"
CURRENT_FUNCTIONAL_TENSOR_MODULE = "torchvision.transforms._functional_tensor"


def _validate_rgb_to_grayscale(rgb_to_grayscale):
    import torch

    sample = torch.tensor(
        [[[[1.0]], [[0.5]], [[0.25]]]],
        dtype=torch.float32,
    )
    actual = rgb_to_grayscale(sample, num_output_channels=1)
    expected = (
        0.2989 * sample[:, 0:1]
        + 0.587 * sample[:, 1:2]
        + 0.114 * sample[:, 2:3]
    )
    if actual.shape != (1, 1, 1, 1):
        raise RuntimeError(
            "torchvision rgb_to_grayscale compatibility check returned "
            f"shape={tuple(actual.shape)}, expected=(1, 1, 1, 1)"
        )
    if actual.dtype != sample.dtype or not torch.allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError(
            "torchvision rgb_to_grayscale compatibility check failed its "
            "reference-value comparison"
        )


def install_functional_tensor_compat():
    """Provide BasicSR 1.4.2's removed functional_tensor import when needed.

    torchvision 0.20 exposes ``rgb_to_grayscale`` from
    ``torchvision.transforms.functional`` but no longer ships the historical
    ``functional_tensor`` module imported by BasicSR 1.4.2.
    """
    try:
        importlib.import_module(LEGACY_FUNCTIONAL_TENSOR_MODULE)
        return False
    except ModuleNotFoundError as exc:
        if exc.name not in {None, LEGACY_FUNCTIONAL_TENSOR_MODULE}:
            raise

    functional_tensor = importlib.import_module(
        CURRENT_FUNCTIONAL_TENSOR_MODULE
    )
    rgb_to_grayscale = getattr(
        functional_tensor,
        "rgb_to_grayscale",
        None,
    )
    if rgb_to_grayscale is None:
        raise ImportError(
            "Installed torchvision does not expose "
            f"{CURRENT_FUNCTIONAL_TENSOR_MODULE}.rgb_to_grayscale"
        )
    _validate_rgb_to_grayscale(rgb_to_grayscale)

    compatibility_module = types.ModuleType(LEGACY_FUNCTIONAL_TENSOR_MODULE)
    compatibility_module.__package__ = "torchvision.transforms"
    compatibility_module.__all__ = ["rgb_to_grayscale"]
    compatibility_module.rgb_to_grayscale = rgb_to_grayscale
    sys.modules.setdefault(
        LEGACY_FUNCTIONAL_TENSOR_MODULE,
        compatibility_module,
    )
    return True

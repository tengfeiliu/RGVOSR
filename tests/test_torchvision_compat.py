import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from dataloaders import torchvision_compat


class TorchvisionCompatTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop(
            torchvision_compat.LEGACY_FUNCTIONAL_TENSOR_MODULE,
            None,
        )

    def test_installs_rgb_to_grayscale_alias_when_legacy_module_is_missing(self):
        sentinel = mock.Mock()

        def fake_import(name):
            if name == torchvision_compat.LEGACY_FUNCTIONAL_TENSOR_MODULE:
                error = ModuleNotFoundError(name)
                error.name = name
                raise error
            if name == torchvision_compat.CURRENT_FUNCTIONAL_TENSOR_MODULE:
                return SimpleNamespace(rgb_to_grayscale=sentinel)
            raise AssertionError(f"Unexpected import: {name}")

        with (
            mock.patch.object(
                torchvision_compat.importlib,
                "import_module",
                side_effect=fake_import,
            ),
            mock.patch.object(
                torchvision_compat,
                "_validate_rgb_to_grayscale",
            ) as validate,
        ):
            installed = (
                torchvision_compat.install_functional_tensor_compat()
            )

        self.assertTrue(installed)
        validate.assert_called_once_with(sentinel)
        compatibility_module = sys.modules[
            torchvision_compat.LEGACY_FUNCTIONAL_TENSOR_MODULE
        ]
        self.assertIs(compatibility_module.rgb_to_grayscale, sentinel)

    def test_does_nothing_when_legacy_module_exists(self):
        existing = object()
        with mock.patch.object(
            torchvision_compat.importlib,
            "import_module",
            return_value=existing,
        ):
            installed = torchvision_compat.install_functional_tensor_compat()

        self.assertFalse(installed)

    def test_does_not_hide_an_unrelated_torchvision_import_failure(self):
        error = ModuleNotFoundError("missing CUDA extension")
        error.name = "torchvision._C"
        with mock.patch.object(
            torchvision_compat.importlib,
            "import_module",
            side_effect=error,
        ):
            with self.assertRaises(ModuleNotFoundError) as raised:
                torchvision_compat.install_functional_tensor_compat()

        self.assertIs(raised.exception, error)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch is required for the numerical compatibility check",
    )
    def test_reference_validation_rejects_wrong_grayscale_math(self):
        import torch

        def wrong_grayscale(image, num_output_channels=1):
            return torch.zeros_like(image[:, :1])

        with self.assertRaisesRegex(
            RuntimeError,
            "reference-value comparison",
        ):
            torchvision_compat._validate_rgb_to_grayscale(
                wrong_grayscale
            )


if __name__ == "__main__":
    unittest.main()

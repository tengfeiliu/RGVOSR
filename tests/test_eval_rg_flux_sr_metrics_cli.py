import ast
import argparse
import json
import tempfile
import unittest
from pathlib import Path


class EvalRgFluxSrMetricsCliTests(unittest.TestCase):
    def test_cli_uses_shared_rg_sr_metrics_module(self):
        source = Path("eval_rg_flux_sr_metrics.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        imports_shared_module = False
        parser_has_dataset_dirs = False
        parser_has_inference_manifest = False
        parser_has_output_dir = False
        parser_has_expected_counts = False
        calls_evaluate_dataset_dirs = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "metrics.rg_sr_metrics":
                imported_names = {alias.name for alias in node.names}
                imports_shared_module = "evaluate_dataset_dirs" in imported_names
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                    args = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
                    parser_has_dataset_dirs = parser_has_dataset_dirs or "--dataset_dirs" in args
                    parser_has_inference_manifest = parser_has_inference_manifest or "--inference_manifest" in args
                    parser_has_output_dir = parser_has_output_dir or "--output_dir" in args
                    parser_has_expected_counts = parser_has_expected_counts or "--expected_counts" in args
                if isinstance(func, ast.Name) and func.id == "evaluate_dataset_dirs":
                    calls_evaluate_dataset_dirs = True

        self.assertTrue(imports_shared_module)
        self.assertTrue(parser_has_dataset_dirs)
        self.assertTrue(parser_has_inference_manifest)
        self.assertTrue(parser_has_output_dir)
        self.assertTrue(parser_has_expected_counts)
        self.assertTrue(calls_evaluate_dataset_dirs)

    def test_inference_manifest_resolves_dataset_dirs_and_default_metrics_output(self):
        source = Path("eval_rg_flux_sr_metrics.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {"load_inference_manifest", "resolve_evaluation_inputs"}
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helpers}, helper_names)

        def fake_parse_name_path(values, flag_name):
            parsed = {}
            for value in values:
                name, raw_path = value.split("=", 1)
                parsed[name] = Path(raw_path)
            return parsed

        namespace = {
            "json": json,
            "Path": Path,
            "parse_name_path": fake_parse_name_path,
        }
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "eval_rg_flux_sr_metrics.py", "exec"), namespace)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inference_root = root / "eval" / "inference" / "run" / "checkpoint-00032000"
            manifest_path = inference_root / "inference_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "output_dir": str(inference_root),
                        "datasets": [
                            {"name": "realLQ250", "output_dir": str(inference_root / "realLQ250")},
                            {"name": "realLR200", "output_dir": str(inference_root / "realLR200")},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            args = argparse.Namespace(
                inference_manifest=manifest_path,
                dataset_dirs=None,
                output_dir=None,
            )
            dataset_dirs, output_dir = namespace["resolve_evaluation_inputs"](args)
            self.assertEqual(dataset_dirs["realLQ250"], inference_root / "realLQ250")
            self.assertEqual(dataset_dirs["realLR200"], inference_root / "realLR200")
            self.assertEqual(output_dir, inference_root / "metrics")

            override = argparse.Namespace(
                inference_manifest=manifest_path,
                dataset_dirs=None,
                output_dir=root / "custom_metrics",
            )
            _, custom_output = namespace["resolve_evaluation_inputs"](override)
            self.assertEqual(custom_output, root / "custom_metrics")

            legacy = argparse.Namespace(
                inference_manifest=None,
                dataset_dirs=["a=/tmp/a", "b=/tmp/b"],
                output_dir=root / "legacy_metrics",
            )
            legacy_dirs, legacy_output = namespace["resolve_evaluation_inputs"](legacy)
            self.assertEqual(legacy_dirs, {"a": Path("/tmp/a"), "b": Path("/tmp/b")})
            self.assertEqual(legacy_output, root / "legacy_metrics")


if __name__ == "__main__":
    unittest.main()

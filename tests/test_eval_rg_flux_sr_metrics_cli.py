import ast
import unittest
from pathlib import Path


class EvalRgFluxSrMetricsCliTests(unittest.TestCase):
    def test_cli_uses_shared_rg_sr_metrics_module(self):
        source = Path("eval_rg_flux_sr_metrics.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        imports_shared_module = False
        parser_has_dataset_dirs = False
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
                    parser_has_output_dir = parser_has_output_dir or "--output_dir" in args
                    parser_has_expected_counts = parser_has_expected_counts or "--expected_counts" in args
                if isinstance(func, ast.Name) and func.id == "evaluate_dataset_dirs":
                    calls_evaluate_dataset_dirs = True

        self.assertTrue(imports_shared_module)
        self.assertTrue(parser_has_dataset_dirs)
        self.assertTrue(parser_has_output_dir)
        self.assertTrue(parser_has_expected_counts)
        self.assertTrue(calls_evaluate_dataset_dirs)


if __name__ == "__main__":
    unittest.main()

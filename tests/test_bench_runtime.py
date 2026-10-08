"""Regression checks for notebook subprocess diagnostics and WDMamba routing."""

import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock


NOTEBOOK = Path(__file__).resolve().parents[1] / "Bench.ipynb"


def notebook_function(name, namespace):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                module = ast.Module(body=[node], type_ignores=[])
                exec(compile(module, str(NOTEBOOK), "exec"), namespace)
                return namespace[name]
    raise AssertionError(f"Missing notebook function: {name}")


class BenchRuntimeTests(unittest.TestCase):
    def test_failed_child_traceback_is_visible_and_retained(self):
        run_command = notebook_function("run_command", {"subprocess": subprocess})
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError) as raised:
            run_command([
                sys.executable, "-u", "-c",
                "print('loading WDMamba', flush=True); raise ModuleNotFoundError('missing_wdmamba_dependency')",
            ])
        for text in ("loading WDMamba", "Traceback", "ModuleNotFoundError: missing_wdmamba_dependency"):
            self.assertIn(text, output.getvalue())
            self.assertIn(text, raised.exception.output)

    def test_successful_child_output_and_working_directory(self):
        run_command = notebook_function("run_command", {"subprocess": subprocess})
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()) as output:
            run_command([sys.executable, "-u", "-c", "import os; print(os.getcwd())"], cwd=Path(temporary))
            self.assertIn(Path(temporary).name.lower(), output.getvalue().lower())

    def test_wdmamba_checks_and_infers_with_selected_interpreter(self):
        root = Path("/project")
        checkpoint = root / "NH_20.83.pth"
        commands = Mock()
        namespace = {
            "ROOT": root, "OUTPUT_ROOT": root / "results", "DATA_ROOT": root / "data",
            "WDMAMBA_PYTHON": "/dedicated-runtime/bin/python", "WDMAMBA_WEIGHT_FILES": {},
            "WDMAMBA_CROSS_DATASET_WEIGHTS": {"i-haze": "nh-haze"},
            "WDMAMBA_PAIRED_ROOTS": {}, "WDMAMBA_LIMIT": 1, "WDMAMBA_MAX_SIDE": 512,
            "SPLIT": "test", "DEVICE": "cuda", "SAVE_IMAGES": True,
            "wdmamba_checkpoint_for": Mock(return_value=checkpoint),
            "validate_dataset": Mock(), "run_command": commands,
        }
        run = notebook_function("run_wdmamba_dataset", namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            run("i-haze", allow_cross=True)
        check, inference = [call.args[0] for call in commands.call_args_list]
        self.assertEqual(check[0], namespace["WDMAMBA_PYTHON"])
        self.assertIn("--check-environment", check)
        self.assertEqual(inference[0], namespace["WDMAMBA_PYTHON"])
        self.assertIn("--save-images", inference)
        self.assertIn("cross-dataset", inference)
        # A failed preflight must prevent any checkpoint inference.
        commands.reset_mock()
        commands.side_effect = subprocess.CalledProcessError(1, check)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            run("i-haze", allow_cross=True)
        self.assertEqual(commands.call_count, 1)


if __name__ == "__main__":
    unittest.main()

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
import zipfile
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
            "SPLIT": "train", "DEVICE": "cuda", "SAVE_IMAGES": True,
            "wdmamba_checkpoint_for": Mock(return_value=checkpoint),
            # Simulate an old helper still present in a live notebook kernel.
            "validate_dataset": lambda dataset: None,
            "validate_wdmamba_dataset": Mock(), "run_command": commands,
        }
        run = notebook_function("run_wdmamba_dataset", namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            run("i-haze", allow_cross=True, split="test", limit=0, max_side=0)
        check, inference = [call.args[0] for call in commands.call_args_list]
        self.assertEqual(check[0], namespace["WDMAMBA_PYTHON"])
        self.assertIn("--check-environment", check)
        self.assertEqual(inference[0], namespace["WDMAMBA_PYTHON"])
        self.assertIn("--save-images", inference)
        self.assertIn("cross-dataset", inference)
        self.assertEqual(inference[inference.index("--split") + 1], "test")
        self.assertEqual(inference[inference.index("--limit") + 1], 0)
        self.assertEqual(inference[inference.index("--max-side") + 1], 0)
        namespace["validate_wdmamba_dataset"].assert_called_once_with("i-haze", "test")
        self.assertEqual(namespace["SPLIT"], "train")
        self.assertEqual(namespace["WDMAMBA_LIMIT"], 1)
        # A failed preflight must prevent any checkpoint inference.
        commands.reset_mock()
        commands.side_effect = subprocess.CalledProcessError(1, check)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            run("i-haze", allow_cross=True, split="test", limit=0, max_side=0)
        self.assertEqual(commands.call_count, 1)

    def test_wdmamba_validation_uses_requested_split_in_existing_kernel(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            root = data / "I-HAZE" / "test"
            for name in ("clear", "hazy"):
                folder = root / name
                folder.mkdir(parents=True)
                (folder / "one.png").touch()
            validate = notebook_function("validate_wdmamba_dataset", {
                "DATA_ROOT": data, "SPLIT": "train",
                "DATASET_LAYOUT": {"i-haze": data / "I-HAZE" / "train"},
            })
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(validate("i-haze", "test"), root)
                with self.assertRaisesRegex(FileNotFoundError, "Missing i-haze train"):
                    validate("i-haze", "train")

    def test_test_suite_runs_all_images_and_packages_twenty_reports(self):
        datasets = ["i-haze", "o-hazy", "sots-indoor", "sots-outdoor", "cdd11"]
        names = ["wdmamba_dataset.xlsx", "wdmamba_domain.xlsx", "wdmamba_fog.xlsx", "hardware.xlsx"]
        for failed_dataset in (None, "o-hazy"):
            with self.subTest(failed_dataset=failed_dataset), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)

                def infer(dataset, **kwargs):
                    if dataset == failed_dataset:
                        raise subprocess.CalledProcessError(1, ["inference", dataset])
                    output = root / "wdmamba" / dataset / "checkpoint"
                    output.mkdir(parents=True)
                    for name in names:
                        (output / name).write_bytes(b"report")
                    return output

                inference = Mock(side_effect=infer)
                run = notebook_function("run_wdmamba_test_suite", {
                    "OUTPUT_ROOT": root, "run_wdmamba_dataset": inference, "subprocess": subprocess,
                })
                with contextlib.redirect_stdout(io.StringIO()):
                    if failed_dataset:
                        with self.assertRaisesRegex(RuntimeError, "Incomplete WDMamba test suite"):
                            run(datasets, max_side=512)
                    else:
                        run(datasets, max_side=512)
                self.assertEqual([call.args[0] for call in inference.call_args_list], datasets)
                for call in inference.call_args_list:
                    self.assertEqual(call.kwargs, {
                        "allow_cross": True, "split": "test", "limit": 0, "max_side": 512,
                    })
                suite = root / "wdmamba" / "test_suite"
                index = json.loads((suite / "reports.json").read_text(encoding="utf-8"))
                completed = [dataset for dataset in datasets if dataset != failed_dataset]
                self.assertEqual(set(index["reports"]), set(completed))
                self.assertEqual(set(index["failures"]), {failed_dataset} if failed_dataset else set())
                with zipfile.ZipFile(suite / "wdmamba_test_reports.zip") as archive:
                    expected = {f"{dataset}/checkpoint/{name}" for dataset in completed for name in names}
                    self.assertEqual(set(archive.namelist()), expected | {"reports.json"})

    def test_resume_keeps_four_completed_datasets_and_runs_cdd11(self):
        datasets = ["i-haze", "o-hazy", "sots-indoor", "sots-outdoor", "cdd11"]
        names = ["wdmamba_dataset.xlsx", "wdmamba_domain.xlsx", "wdmamba_fog.xlsx", "hardware.xlsx"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"

            def write_reports(dataset, **kwargs):
                output = root / "wdmamba" / dataset / checkpoint.stem
                output.mkdir(parents=True, exist_ok=True)
                for name in names:
                    (output / name).write_bytes(b"report")
                (output / "run_config.json").write_text(json.dumps({
                    "dataset": dataset, "checkpoint": str(checkpoint), "split": "test",
                    "limit": 0, "max_side": 512, "images": 1,
                }), encoding="utf-8")
                return output

            for dataset in datasets[:-1]:
                write_reports(dataset)
            inference = Mock(side_effect=write_reports)
            run = notebook_function("run_wdmamba_test_suite", {
                "OUTPUT_ROOT": root, "run_wdmamba_dataset": inference, "subprocess": subprocess,
                "wdmamba_checkpoint_for": Mock(return_value=checkpoint),
            })
            with contextlib.redirect_stdout(io.StringIO()):
                zip_path = run(datasets, max_side=512, reuse_completed=True)
            inference.assert_called_once_with("cdd11", allow_cross=True, split="test", limit=0, max_side=512)
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(len([name for name in archive.namelist() if name.endswith(".xlsx")]), 20)
            # Smoke-test results must not be reused as complete test results.
            config_path = root / "wdmamba" / "i-haze" / checkpoint.stem / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["limit"] = 1
            config_path.write_text(json.dumps(config), encoding="utf-8")
            inference.reset_mock()
            with contextlib.redirect_stdout(io.StringIO()):
                run(datasets, max_side=512, reuse_completed=True)
            inference.assert_called_once_with("i-haze", allow_cross=True, split="test", limit=0, max_side=512)


if __name__ == "__main__":
    unittest.main()

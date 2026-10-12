"""Download and normalize benchmark checkpoints used by ``Bench.ipynb``.

The upstream projects publish weights in Google Drive folders.  This helper is
deliberately resumable: existing non-empty files are kept, while missing files
are downloaded and copied to the paths expected by the evaluators.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import subprocess
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# These IDs are the folders used by commit 6d50a98.  Keep them in one place so
# a notebook rerun cannot silently drift to a different checkpoint source.
DRIVE_FOLDERS = {
    "udpnet": "1_XDXqcdqF5DuuZSGQQC4Je_5SZer7-Jv",
    "mb_taylorformerv2": "1_85eRPM2ZMU74BEziHFXktp2um1ueUoC",
    "hazewavenet": "1DR5JvR43JcO1goxL-sc4vuodl-ZiOJ1P",
}

DIRECT_FILES = {
    "griddehazenet": (
        ("indoor_haze_best_3_6", "https://raw.githubusercontent.com/Darkin72/GridDehazeNet/main/indoor_haze_best_3_6"),
        ("outdoor_haze_best_3_6", "https://raw.githubusercontent.com/Darkin72/GridDehazeNet/main/outdoor_haze_best_3_6"),
    ),
}
GRID_REPOSITORY = "https://github.com/Darkin72/GridDehazeNet.git"


def ensure_grid_source() -> Path:
    """Ensure the evaluator source exists even when the checkout has weights only."""
    target = PROJECT_ROOT / "solution" / "GridDehazeNet"
    required = (target / "model.py", target / "residual_dense_block.py")
    if all(path.is_file() for path in required):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="griddehazenet-") as temporary:
        staging = Path(temporary) / "GridDehazeNet"
        subprocess.run(
            ["git", "clone", "--depth", "1", GRID_REPOSITORY, str(staging)],
            check=True,
        )
        shutil.copytree(staging, target, dirs_exist_ok=True)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("GridDehazeNet source is incomplete: " + ", ".join(missing))
    return target

REQUIRED = {
    "udpnet": (
        ("FSNet_UDPNet_ITS.ckpt", PROJECT_ROOT / "UDPNet_checkpoints"),
        ("FSNet_UDPNet_OTS.ckpt", PROJECT_ROOT / "UDPNet_checkpoints"),
    ),
    "mb_taylorformerv2": (
        ("ITS-B.pth", PROJECT_ROOT / "mb-taylorformerv2"),
        ("OTS-B.pth", PROJECT_ROOT / "mb-taylorformerv2"),
    ),
    "hazewavenet": tuple(
        (name, PROJECT_ROOT / "checkpoints/haze_wavelet")
        for name in (
            "haze_wavelet_i_haze.pt",
            "haze_wavelet_o_hazy.pt",
            "haze_wavelet_sots_its.pt",
        )
    ),
}


def _download_folder(folder_id: str, destination: Path) -> list[Path]:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required; run `pip install gdown>=5`.") from exc
    destination.mkdir(parents=True, exist_ok=True)
    # download_folder handles nested files and resumes files that already exist.
    gdown.download_folder(
        id=folder_id,
        output=str(destination),
        quiet=False,
        use_cookies=False,
    )
    return [path for path in destination.rglob("*") if path.is_file()]


def _find(files: list[Path], name: str) -> Path | None:
    exact = [path for path in files if path.name == name]
    return exact[0] if exact else None


def download(kind: str, cache_root: Path) -> list[Path]:
    if kind in DIRECT_FILES:
        ensure_grid_source()
        paths = []
        target_root = PROJECT_ROOT / "solution" / "GridDehazeNet"
        target_root.mkdir(parents=True, exist_ok=True)
        for name, url in DIRECT_FILES[kind]:
            target = target_root / name
            if not target.is_file() or target.stat().st_size == 0:
                partial = target.with_suffix(target.suffix + ".part")
                print(f"  downloading: {url}", flush=True)
                urllib.request.urlretrieve(url, partial)
                partial.replace(target)
            paths.append(target)
        return paths
    if kind not in DRIVE_FOLDERS:
        raise ValueError(f"Unknown checkpoint group: {kind}")
    files = _download_folder(DRIVE_FOLDERS[kind], cache_root / kind)
    copied: list[Path] = []
    missing: list[str] = []
    for name, target_dir in REQUIRED[kind]:
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        if target.is_file() and target.stat().st_size > 0:
            copied.append(target)
            continue
        source = _find(files, name)
        if source is None:
            missing.append(name)
            continue
        shutil.copy2(source, target)
        copied.append(target)
    if missing:
        raise FileNotFoundError(
            f"{kind}: missing files {missing}. Downloaded files were under {cache_root / kind}"
        )
    if kind == "hazewavenet":
        optional = target_dir / "haze_wavelet_sots_ots.pt"
        if not optional.is_file():
            print("  note: sots_ots checkpoint is not published in the commit's Drive folder; "
                  "the SOTS-outdoor run will be skipped unless it is supplied manually.", flush=True)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", *DRIVE_FOLDERS, *DIRECT_FILES), default="all")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "checkpoints" / "downloads")
    args = parser.parse_args()
    groups = [*DRIVE_FOLDERS, *DIRECT_FILES] if args.group == "all" else [args.group]
    for group in groups:
        print(f"[{group}] downloading/checking checkpoint folder", flush=True)
        for path in download(group, args.cache_root):
            print(f"  ready: {path}", flush=True)
    # Report tracked/local checkpoints explicitly so a run fails before
    # inference if a checkout is incomplete.
    tracked = (
        PROJECT_ROOT / "solution/GridDehazeNet/indoor_haze_best_3_6",
        PROJECT_ROOT / "solution/GridDehazeNet/outdoor_haze_best_3_6",
    )
    missing = [str(path) for path in tracked if not path.is_file()]
    if missing:
        raise FileNotFoundError("Tracked checkpoints missing: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())

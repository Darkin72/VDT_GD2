"""Evaluate GridDehazeNet on the same four datasets as evaluate.py."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from excel_report import export_excel_reports

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = Path(__file__).resolve().parent / "evaluate.py"

DATASET_GROUPS = (
    ("indoor", "sots-indoor", "i-haze"),
    ("outdoor", "sots-outdoor", "o-hazy"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá GridDehazeNet trên SOTS, O-HAZE và I-HAZE."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "evaluation_results" / "griddehazenet",
    )
    parser.add_argument("--indoor-checkpoint", type=Path,
                        default=PROJECT_ROOT / "solution" / "GridDehazeNet" / "indoor_haze_best_3_6")
    parser.add_argument("--outdoor-checkpoint", type=Path,
                        default=PROJECT_ROOT / "solution" / "GridDehazeNet" / "outdoor_haze_best_3_6")
    parser.add_argument("--device", default=None, help="cpu hoặc cuda; mặc định tự chọn.")
    parser.add_argument("--max-side", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def run_group(args, name, *datasets):
    checkpoint = args.indoor_checkpoint if name == "indoor" else args.outdoor_checkpoint
    output_dir = args.output_dir / name
    command = [
        sys.executable,
        str(EVALUATOR),
        "--solution", "griddehazenet",
        "--dataset", *datasets,
        "--data-root", str(args.data_root),
        "--split", args.split,
        "--output-dir", str(output_dir),
        "--grid-checkpoint", str(checkpoint),
        "--max-side", str(args.max_side),
    ]
    if args.device:
        command.extend(("--grid-device", args.device))
    if args.limit > 0:
        command.extend(("--limit", str(args.limit)))
    if args.save_images:
        command.append("--save-images")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    rows = {}
    for dataset in datasets:
        summary_path = output_dir / "griddehazenet" / args.split / "summary.csv"
        with summary_path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                if row["dataset"] == dataset:
                    rows[dataset] = row
    performance_path = output_dir / "griddehazenet" / args.split / "performance.json"
    with performance_path.open(encoding="utf-8") as file:
        performance = json.load(file)
    per_image_path = output_dir / "griddehazenet" / args.split / "per_image.csv"
    with per_image_path.open(newline="", encoding="utf-8-sig") as file:
        per_image_rows = list(csv.DictReader(file))
    summaries = {}
    for filename in ("origin_summary.csv", "fog_summary.csv"):
        summary_path = output_dir / "griddehazenet" / args.split / filename
        with summary_path.open(newline="", encoding="utf-8-sig") as file:
            summaries[filename] = list(csv.DictReader(file))
    return rows, performance, summaries, per_image_rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    performances = {}
    domain_summaries = {"origin_summary.csv": [], "fog_summary.csv": []}
    all_per_image_rows = []
    for name, *datasets in DATASET_GROUPS:
        rows, performance, summaries, per_image_rows = run_group(args, name, *datasets)
        all_rows.update(rows)
        performances[name] = performance
        all_per_image_rows.extend(per_image_rows)
        for filename, values in summaries.items():
            domain_summaries[filename].extend(values)

    labels = {
        "sots-indoor": "SOTS-Indoor",
        "sots-outdoor": "SOTS-Outdoor",
        "o-hazy": "O-HAZE",
        "i-haze": "I-HAZE",
    }
    output_path = args.output_dir / f"summary_{args.split}.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("Dataset", "PSNR", "SSIM", "ms / ảnh", "Images"))
        for dataset in ("sots-indoor", "sots-outdoor", "o-hazy", "i-haze"):
            row = all_rows[dataset]
            writer.writerow((labels[dataset], row["output_psnr"], row["output_ssim"], row["mean_runtime_ms"], row["images"]))

    for filename, values in domain_summaries.items():
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8-sig") as file:
            if values:
                writer = csv.DictWriter(file, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)

    performance_path = args.output_dir / f"performance_{args.split}.json"
    with performance_path.open("w", encoding="utf-8") as file:
        json.dump(performances, file, ensure_ascii=False, indent=2)
    excel_paths = export_excel_reports(
        "GridDehazeNet", all_per_image_rows, args.output_dir
    )

    print(f"\nĐã lưu bảng tổng hợp tại: {output_path.resolve()}")
    print(f"Đã lưu thông tin hardware/FPS tại: {performance_path.resolve()}")
    print("Đã lưu ba báo cáo Excel:")
    for path in excel_paths.values():
        print(f"- {path.resolve()}")
    print("\nKẾT QUẢ CHÍNH")
    print("| Dataset | PSNR | SSIM | ms / ảnh |")
    print("|---|---:|---:|---:|")
    for dataset in ("sots-indoor", "sots-outdoor", "o-hazy", "i-haze"):
        row = all_rows[dataset]
        print(f"| {labels[dataset]} | {float(row['output_psnr']):.4f} | {float(row['output_ssim']):.4f} | {float(row['mean_runtime_ms']):.2f} |")

    print("\n1. REAL / SYNTHETIC")
    print("| Nguồn | PSNR | SSIM | ms / ảnh |")
    print("|---|---:|---:|---:|")
    for row in domain_summaries["origin_summary.csv"]:
        print(f"| {row['data_origin']} | {float(row['output_psnr']):.4f} | {float(row['output_ssim']):.4f} | {float(row['mean_runtime_ms']):.2f} |")

    print("\n2. FOG LEVEL")
    print("| Dataset | Fog level | PSNR | SSIM | ms / ảnh |")
    print("|---|---|---:|---:|---:|")
    for row in domain_summaries["fog_summary.csv"]:
        print(f"| {row['dataset']} | {row['fog_level']} | {float(row['output_psnr']):.4f} | {float(row['output_ssim']):.4f} | {float(row['mean_runtime_ms']):.2f} |")


if __name__ == "__main__":
    main()

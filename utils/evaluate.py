import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOLUTION_ROOT = PROJECT_ROOT / "solution"
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from DPC.DCP import dehaze as dcp_dehaze


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_files(directory):
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_i_haze(data_root, split):
    base = data_root / "I-HAZE" / split
    clear_lookup = {
        path.stem: path for path in image_files(base / "clear")
    }
    pairs = []
    for hazy_path in image_files(base / "hazy"):
        image_id = hazy_path.stem.removesuffix("_hazy")
        if image_id in clear_lookup:
            pairs.append(
                ("i-haze", image_id, hazy_path, clear_lookup[image_id])
            )
    return pairs


def collect_o_hazy(data_root, split):
    base = data_root / "O-HAZY" / split
    clear_lookup = {
        path.stem.lower(): path for path in image_files(base / "clear")
    }
    pairs = []
    for hazy_path in image_files(base / "hazy"):
        image_id = hazy_path.stem.lower()
        if image_id in clear_lookup:
            pairs.append(
                ("o-hazy", image_id, hazy_path, clear_lookup[image_id])
            )
    return pairs


def collect_sots(data_root, domain, split):
    base = (
        data_root
        / "Synthetic Objective Testing Set (SOTS) [RESIDE]"
        / domain
        / split
    )
    clear_lookup = {
        path.stem: path for path in image_files(base / "clear")
    }
    pairs = []
    for hazy_path in image_files(base / "hazy"):
        image_id = hazy_path.stem.split("_")[0]
        if image_id in clear_lookup:
            pairs.append(
                (
                    f"sots-{domain}",
                    hazy_path.stem,
                    hazy_path,
                    clear_lookup[image_id],
                )
            )
    return pairs


DATASET_LOADERS = {
    "i-haze": collect_i_haze,
    "o-hazy": collect_o_hazy,
    "sots-indoor": lambda root, split: collect_sots(
        root, "indoor", split
    ),
    "sots-outdoor": lambda root, split: collect_sots(
        root, "outdoor", split
    ),
}


def resize_pair(hazy, clear, max_side):
    if hazy.shape[:2] != clear.shape[:2]:
        hazy = cv2.resize(
            hazy,
            (clear.shape[1], clear.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    if max_side <= 0:
        return hazy, clear
    height, width = clear.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return hazy, clear
    size = (round(width * scale), round(height * scale))
    return (
        cv2.resize(hazy, size, interpolation=cv2.INTER_AREA),
        cv2.resize(clear, size, interpolation=cv2.INTER_AREA),
    )


def calculate_psnr(first, second):
    error_sum = 0.0
    value_count = first.size
    stripe_height = 512
    for y0 in range(0, first.shape[0], stripe_height):
        y1 = min(y0 + stripe_height, first.shape[0])
        difference = (
            first[y0:y1].astype(np.float32)
            - second[y0:y1].astype(np.float32)
        )
        error_sum += float(np.sum(difference * difference))
    mse = error_sum / value_count
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0**2) / mse)


def calculate_ssim(first, second):
    first = first.astype(np.float32) / 255.0
    second = second.astype(np.float32) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    window = (11, 11)
    sigma = 1.5
    padding = window[0] // 2
    stripe_height = 512
    score_sum = 0.0
    pixel_count = 0

    for channel in range(3):
        first_channel = first[:, :, channel]
        second_channel = second[:, :, channel]
        for y0 in range(0, first.shape[0], stripe_height):
            y1 = min(y0 + stripe_height, first.shape[0])
            source_y0 = max(0, y0 - padding)
            source_y1 = min(first.shape[0], y1 + padding)
            x = first_channel[source_y0:source_y1]
            y = second_channel[source_y0:source_y1]

            mean_x = cv2.GaussianBlur(
                x, window, sigma, borderType=cv2.BORDER_REFLECT
            )
            mean_y = cv2.GaussianBlur(
                y, window, sigma, borderType=cv2.BORDER_REFLECT
            )
            variance_x = cv2.GaussianBlur(
                x * x, window, sigma, borderType=cv2.BORDER_REFLECT
            ) - mean_x * mean_x
            variance_y = cv2.GaussianBlur(
                y * y, window, sigma, borderType=cv2.BORDER_REFLECT
            ) - mean_y * mean_y
            covariance = cv2.GaussianBlur(
                x * y, window, sigma, borderType=cv2.BORDER_REFLECT
            ) - mean_x * mean_y

            numerator = (
                (2 * mean_x * mean_y + c1)
                * (2 * covariance + c2)
            )
            denominator = (
                (mean_x * mean_x + mean_y * mean_y + c1)
                * (variance_x + variance_y + c2)
            )
            score = numerator / np.maximum(denominator, 1e-12)
            core_y0 = y0 - source_y0
            core_y1 = core_y0 + (y1 - y0)
            core = score[core_y0:core_y1]
            score_sum += float(np.sum(core, dtype=np.float64))
            pixel_count += core.size

    return score_sum / pixel_count


def run_dcp(image, args):
    return dcp_dehaze(
        image,
        patch_size=args.patch_size,
        omega=args.omega,
        t0=args.t0,
        top_percent=args.top_percent,
        use_guided_filter=args.guided_filter,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
    )[0]


SOLUTIONS = {
    "dcp": run_dcp,
}


def finite_mean(values):
    finite_values = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else float("inf")


def summarize(rows):
    summaries = []
    dataset_names = sorted({row["dataset"] for row in rows})
    for dataset_name in dataset_names:
        subset = [row for row in rows if row["dataset"] == dataset_name]
        summaries.append(
            {
                "dataset": dataset_name,
                "images": len(subset),
                "hazy_psnr": finite_mean(
                    [row["hazy_psnr"] for row in subset]
                ),
                "hazy_ssim": float(
                    np.mean([row["hazy_ssim"] for row in subset])
                ),
                "output_psnr": finite_mean(
                    [row["output_psnr"] for row in subset]
                ),
                "output_ssim": float(
                    np.mean([row["output_ssim"] for row in subset])
                ),
                "psnr_improvement": float(
                    np.mean([row["psnr_improvement"] for row in subset])
                ),
                "ssim_improvement": float(
                    np.mean([row["ssim_improvement"] for row in subset])
                ),
                "mean_runtime_ms": float(
                    np.mean([row["runtime_ms"] for row in subset])
                ),
            }
        )
    if len(dataset_names) > 1:
        summaries.append(
            {
                "dataset": "all",
                "images": len(rows),
                "hazy_psnr": finite_mean(
                    [row["hazy_psnr"] for row in rows]
                ),
                "hazy_ssim": float(
                    np.mean([row["hazy_ssim"] for row in rows])
                ),
                "output_psnr": finite_mean(
                    [row["output_psnr"] for row in rows]
                ),
                "output_ssim": float(
                    np.mean([row["output_ssim"] for row in rows])
                ),
                "psnr_improvement": float(
                    np.mean([row["psnr_improvement"] for row in rows])
                ),
                "ssim_improvement": float(
                    np.mean([row["ssim_improvement"] for row in rows])
                ),
                "mean_runtime_ms": float(
                    np.mean([row["runtime_ms"] for row in rows])
                ),
            }
        )
    return summaries


def infer_data_origin(dataset_name):
    return "synthetic" if dataset_name.startswith("sots-") else "real"


def infer_fog_level(dataset_name, image_id):
    if dataset_name not in {"sots-indoor", "sots-outdoor"}:
        return ""
    try:
        level = float(image_id.rsplit("_", 1)[-1])
    except ValueError:
        return ""

    if dataset_name == "sots-indoor":
        if 1 <= level <= 4:
            return "light"
        if 5 <= level <= 8:
            return "medium"
        if 9 <= level <= 10:
            return "heavy"
    else:
        if level <= 0.08:
            return "light"
        if level <= 0.12:
            return "medium"
        return "heavy"
    return ""


def summarize_origins(rows):
    summaries = []
    for data_origin in ("real", "synthetic"):
        subset = [
            row for row in rows if row["data_origin"] == data_origin
        ]
        if not subset:
            continue
        summary = summarize(subset)[-1]
        summary["data_origin"] = data_origin
        summary.pop("dataset")
        summaries.append(summary)
    return summaries


def summarize_fog_levels(rows):
    fog_order = ("light", "medium", "heavy")
    summaries = []
    datasets = ("sots-indoor", "sots-outdoor")
    for dataset_name in datasets:
        for fog_level in fog_order:
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset_name
                and row["fog_level"] == fog_level
            ]
            if not subset:
                continue
            summary = summarize(subset)[0]
            summary["fog_level"] = fog_level
            summaries.append(summary)
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_number(value):
    return "inf" if not math.isfinite(value) else f"{value:.4f}"


def evaluate_pair(pair, solution, args, image_directory):
    dataset_name, image_id, hazy_path, clear_path = pair
    hazy_bgr = cv2.imread(str(hazy_path), cv2.IMREAD_COLOR)
    clear_bgr = cv2.imread(str(clear_path), cv2.IMREAD_COLOR)
    if hazy_bgr is None or clear_bgr is None:
        return None, f"Bỏ qua ảnh không đọc được: {image_id}"

    hazy_rgb = cv2.cvtColor(hazy_bgr, cv2.COLOR_BGR2RGB)
    clear_rgb = cv2.cvtColor(clear_bgr, cv2.COLOR_BGR2RGB)
    hazy_rgb, clear_rgb = resize_pair(hazy_rgb, clear_rgb, args.max_side)

    start_time = time.perf_counter()
    output_rgb = solution(hazy_rgb, args)
    runtime_ms = (time.perf_counter() - start_time) * 1000

    hazy_psnr = calculate_psnr(hazy_rgb, clear_rgb)
    hazy_ssim = calculate_ssim(hazy_rgb, clear_rgb)
    output_psnr = calculate_psnr(output_rgb, clear_rgb)
    output_ssim = calculate_ssim(output_rgb, clear_rgb)
    row = {
        "dataset": dataset_name,
        "image_id": image_id,
        "data_origin": infer_data_origin(dataset_name),
        "fog_level": infer_fog_level(dataset_name, image_id),
        "width": clear_rgb.shape[1],
        "height": clear_rgb.shape[0],
        "hazy_psnr": hazy_psnr,
        "hazy_ssim": hazy_ssim,
        "output_psnr": output_psnr,
        "output_ssim": output_ssim,
        "psnr_improvement": output_psnr - hazy_psnr,
        "ssim_improvement": output_ssim - hazy_ssim,
        "runtime_ms": runtime_ms,
        "hazy_path": str(hazy_path),
        "clear_path": str(clear_path),
    }

    if args.save_images:
        output_path = image_directory / f"{dataset_name}_{image_id}_{args.solution}.png"
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR),
        )
    return row, None


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Đánh giá tự động các thuật toán khử sương."
    )
    parser.add_argument(
        "--solution",
        required=True,
        choices=sorted(SOLUTIONS),
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        choices=["all", *DATASET_LOADERS],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "dataset",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Tập dữ liệu cần đánh giá (mặc định: test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "evaluation_results",
    )
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Số ảnh xử lý đồng thời.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="0 giữ nguyên độ phân giải; giá trị dương resize cạnh dài nhất.",
    )
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--omega", type=float, default=0.95)
    parser.add_argument("--t0", type=float, default=0.1)
    parser.add_argument("--top-percent", type=float, default=0.001)
    parser.add_argument(
        "--guided-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--guided-radius", type=int, default=40)
    parser.add_argument("--guided-eps", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.concurrent < 1:
        raise ValueError("--concurrent phải lớn hơn hoặc bằng 1.")
    selected_datasets = (
        list(DATASET_LOADERS)
        if "all" in args.dataset
        else list(dict.fromkeys(args.dataset))
    )
    pairs = []
    for dataset_name in selected_datasets:
        dataset_pairs = DATASET_LOADERS[dataset_name](
            args.data_root,
            args.split,
        )
        if args.limit > 0:
            dataset_pairs = dataset_pairs[: args.limit]
        pairs.extend(dataset_pairs)

    if not pairs:
        raise RuntimeError(
            f"Không tìm thấy cặp ảnh nào trong split {args.split}. "
            "Kiểm tra --data-root và --split."
        )

    run_directory = args.output_dir / args.solution / args.split
    image_directory = run_directory / "images"
    run_directory.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        image_directory.mkdir(parents=True, exist_ok=True)

    total = len(pairs)
    progress = tqdm(
        total=total,
        desc=f"Đánh giá {args.solution}",
        unit="ảnh",
    )
    solution = SOLUTIONS[args.solution]
    rows = []
    if args.concurrent == 1:
        completed = (
            evaluate_pair(pair, solution, args, image_directory)
            for pair in pairs
        )
        for row, error in completed:
            progress.update(1)
            if error:
                progress.write(error)
                continue
            rows.append(row)
            progress.set_postfix(
                dataset=row["dataset"],
                psnr=f"{row['output_psnr']:.2f}",
                ssim=f"{row['output_ssim']:.3f}",
                refresh=False,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            futures = [
                executor.submit(
                    evaluate_pair,
                    pair,
                    solution,
                    args,
                    image_directory,
                )
                for pair in pairs
            ]
            for future in as_completed(futures):
                row, error = future.result()
                progress.update(1)
                if error:
                    progress.write(error)
                    continue
                rows.append(row)
                progress.set_postfix(
                    dataset=row["dataset"],
                    psnr=f"{row['output_psnr']:.2f}",
                    ssim=f"{row['output_ssim']:.3f}",
                    refresh=False,
                )
    progress.close()

    rows.sort(key=lambda row: (row["dataset"], row["image_id"]))
    summaries = summarize(rows)
    origin_summaries = summarize_origins(rows)
    fog_summaries = summarize_fog_levels(rows)
    write_csv(run_directory / "per_image.csv", rows)
    write_csv(run_directory / "summary.csv", summaries)
    write_csv(run_directory / "origin_summary.csv", origin_summaries)
    write_csv(run_directory / "fog_summary.csv", fog_summaries)

    configuration = {
        "solution": args.solution,
        "datasets": selected_datasets,
        "split": args.split,
        "data_root": str(args.data_root.resolve()),
        "max_side": args.max_side,
        "save_images": args.save_images,
        "concurrent": args.concurrent,
        "fog_levels": {
            "sots_indoor": {
                "light": "hậu tố _1 đến _4",
                "medium": "hậu tố _5 đến _8",
                "heavy": "hậu tố _9 đến _10",
            },
            "sots_outdoor": {
                "light": "beta = 0.08",
                "medium": "beta = 0.12",
                "heavy": "beta = 0.16 hoặc 0.20",
            },
            "real_datasets": "I-HAZE và O-HAZY không có nhãn mức sương.",
        },
        "parameters": {
            "patch_size": args.patch_size,
            "omega": args.omega,
            "t0": args.t0,
            "top_percent": args.top_percent,
            "guided_filter": args.guided_filter,
            "guided_radius": args.guided_radius,
            "guided_eps": args.guided_eps,
        },
    }
    with (run_directory / "config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(configuration, file, ensure_ascii=False, indent=2)

    print("\nKẾT QUẢ TỔNG HỢP")
    print(
        f"{'Dataset':<16} {'N':>5} {'PSNR':>10} {'SSIM':>10} "
        f"{'ΔPSNR':>10} {'ΔSSIM':>10} {'ms/ảnh':>12}"
    )
    for summary in summaries:
        print(
            f"{summary['dataset']:<16} {summary['images']:>5} "
            f"{format_number(summary['output_psnr']):>10} "
            f"{format_number(summary['output_ssim']):>10} "
            f"{format_number(summary['psnr_improvement']):>10} "
            f"{format_number(summary['ssim_improvement']):>10} "
            f"{summary['mean_runtime_ms']:>12.2f}"
        )

    if origin_summaries:
        print("\nKẾT QUẢ THEO NGUỒN DỮ LIỆU")
        print(
            f"{'Nguồn':<16} {'N':>5} {'PSNR':>10} {'SSIM':>10} "
            f"{'ΔPSNR':>10} {'ΔSSIM':>10} {'ms/ảnh':>12}"
        )
        for summary in origin_summaries:
            print(
                f"{summary['data_origin']:<16} {summary['images']:>5} "
                f"{format_number(summary['output_psnr']):>10} "
                f"{format_number(summary['output_ssim']):>10} "
                f"{format_number(summary['psnr_improvement']):>10} "
                f"{format_number(summary['ssim_improvement']):>10} "
                f"{summary['mean_runtime_ms']:>12.2f}"
            )

    if fog_summaries:
        print("\nKẾT QUẢ THEO MỨC SƯƠNG SOTS")
        print(
            f"{'Dataset':<16} {'Mức':<8} {'N':>5} {'PSNR':>10} "
            f"{'SSIM':>10} {'ΔPSNR':>10} {'ΔSSIM':>10} {'ms/ảnh':>12}"
        )
        for summary in fog_summaries:
            print(
                f"{summary['dataset']:<16} "
                f"{summary['fog_level']:<8} "
                f"{summary['images']:>5} "
                f"{format_number(summary['output_psnr']):>10} "
                f"{format_number(summary['output_ssim']):>10} "
                f"{format_number(summary['psnr_improvement']):>10} "
                f"{format_number(summary['ssim_improvement']):>10} "
                f"{summary['mean_runtime_ms']:>12.2f}"
            )
    print(f"\nĐã lưu kết quả tại: {run_directory.resolve()}")


if __name__ == "__main__":
    main()

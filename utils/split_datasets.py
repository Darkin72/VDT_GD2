import argparse
import ast
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_NAMES = ("train", "val", "test")
SPLIT_WEIGHTS = (5, 1, 1)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def allocate_split_counts(item_count):
    total_weight = sum(SPLIT_WEIGHTS)
    exact_counts = [
        item_count * weight / total_weight for weight in SPLIT_WEIGHTS
    ]
    counts = [int(count) for count in exact_counts]
    remainder = item_count - sum(counts)

    # Khi phần dư bằng nhau, ưu tiên test trước val để tập test không nhỏ hơn.
    tie_priority = {"train": 0, "test": 1, "val": 2}
    ranked_indices = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (
            -(exact_counts[index] - counts[index]),
            tie_priority[SPLIT_NAMES[index]],
        ),
    )
    for index in ranked_indices[:remainder]:
        counts[index] += 1
    return dict(zip(SPLIT_NAMES, counts))


def split_ids(image_ids):
    ordered_ids = sorted(image_ids, key=natural_key)
    counts = allocate_split_counts(len(ordered_ids))
    result = {}
    start = 0
    for split_name in SPLIT_NAMES:
        end = start + counts[split_name]
        result[split_name] = ordered_ids[start:end]
        start = end
    return result


def collect_images(directories):
    images = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = path.stem.lower()
            if key in images and images[key].resolve() != path.resolve():
                raise RuntimeError(
                    f"Trùng ID ảnh {key}: {images[key]} và {path}"
                )
            images[key] = path
    return images


def move_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        raise FileExistsError(f"Tệp đích đã tồn tại: {destination}")
    shutil.move(str(source), str(destination))


def remove_empty_directories(directories):
    for directory in directories:
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()


def split_o_hazy(data_root):
    base = data_root / "O-HAZY"
    clear_directories = [base / "GT", base / "clear"]
    hazy_directories = [base / "hazy"]
    for split_name in SPLIT_NAMES:
        clear_directories.append(base / split_name / "clear")
        hazy_directories.append(base / split_name / "hazy")

    clear_images = collect_images(clear_directories)
    hazy_images = collect_images(hazy_directories)
    if set(clear_images) != set(hazy_images):
        raise RuntimeError(
            "O-HAZY không khớp ID giữa ảnh clear và hazy: "
            f"thiếu clear={sorted(set(hazy_images) - set(clear_images))}, "
            f"thiếu hazy={sorted(set(clear_images) - set(hazy_images))}"
        )

    assignments = split_ids(clear_images)
    for split_name, image_ids in assignments.items():
        for image_id in image_ids:
            move_image(
                clear_images[image_id],
                base / split_name / "clear" / clear_images[image_id].name,
            )
            move_image(
                hazy_images[image_id],
                base / split_name / "hazy" / hazy_images[image_id].name,
            )

    remove_empty_directories([base / "GT", base / "clear", base / "hazy"])
    return {name: len(ids) for name, ids in assignments.items()}


def collect_sots_domain(domain_root):
    clear_directories = [domain_root / "clear"]
    hazy_directories = [domain_root / "hazy"]
    for split_name in SPLIT_NAMES:
        clear_directories.append(domain_root / split_name / "clear")
        hazy_directories.append(domain_root / split_name / "hazy")

    clear_images = collect_images(clear_directories)
    hazy_images = collect_images(hazy_directories)
    hazy_groups = defaultdict(list)
    for path in hazy_images.values():
        image_id = path.stem.split("_", 1)[0].lower()
        hazy_groups[image_id].append(path)

    if set(clear_images) != set(hazy_groups):
        raise RuntimeError(
            f"{domain_root.name} không khớp ID giữa ảnh clear và hazy: "
            f"thiếu clear={sorted(set(hazy_groups) - set(clear_images))}, "
            f"thiếu hazy={sorted(set(clear_images) - set(hazy_groups))}"
        )
    for paths in hazy_groups.values():
        paths.sort(key=lambda path: natural_key(path.name))
    return clear_images, hazy_groups


def split_sots_domain(domain_root):
    clear_images, hazy_groups = collect_sots_domain(domain_root)
    assignments = split_ids(clear_images)
    image_to_split = {}

    for split_name, image_ids in assignments.items():
        for image_id in image_ids:
            image_to_split[image_id] = split_name
            clear_path = clear_images[image_id]
            move_image(
                clear_path,
                domain_root / split_name / "clear" / clear_path.name,
            )
            for hazy_path in hazy_groups[image_id]:
                move_image(
                    hazy_path,
                    domain_root / split_name / "hazy" / hazy_path.name,
                )

    remove_empty_directories(
        [domain_root / "clear", domain_root / "hazy"]
    )
    counts = {
        split_name: {
            "clear": len(image_ids),
            "hazy": sum(len(hazy_groups[image_id]) for image_id in image_ids),
        }
        for split_name, image_ids in assignments.items()
    }
    return counts, image_to_split


def read_indoor_metadata(metadata_path):
    with metadata_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required_columns = {
        "image_id",
        "clear_image_path",
        "hazy_image_paths",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(
            f"Metadata không có đủ cột bắt buộc: {metadata_path}"
        )
    return rows


def update_indoor_metadata(metadata_path, image_to_split):
    rows = read_indoor_metadata(metadata_path)
    updated_rows = []
    for row in rows:
        image_id = row["image_id"].lower()
        if image_id not in image_to_split:
            raise RuntimeError(f"Metadata chứa ID không tồn tại: {image_id}")
        split_name = image_to_split[image_id]
        clear_name = Path(row["clear_image_path"]).name
        hazy_names = [
            Path(path).name for path in ast.literal_eval(row["hazy_image_paths"])
        ]
        updated_rows.append(
            {
                "image_id": row["image_id"],
                "split": split_name,
                "clear_image_path": (
                    Path(split_name) / "clear" / clear_name
                ).as_posix(),
                "hazy_image_paths": repr(
                    [
                        (Path(split_name) / "hazy" / name).as_posix()
                        for name in hazy_names
                    ]
                ),
            }
        )

    updated_rows.sort(key=lambda row: natural_key(row["image_id"]))
    with metadata_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "image_id",
                "split",
                "clear_image_path",
                "hazy_image_paths",
            ),
        )
        writer.writeheader()
        writer.writerows(updated_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Chia O-HAZY và SOTS theo tỷ lệ I-HAZE 25:5:5 "
            "(tương đương 5:1:1)."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    sots_root = (
        data_root / "Synthetic Objective Testing Set (SOTS) [RESIDE]"
    )

    o_hazy_counts = split_o_hazy(data_root)
    indoor_counts, indoor_assignments = split_sots_domain(
        sots_root / "indoor"
    )
    outdoor_counts, _ = split_sots_domain(sots_root / "outdoor")
    update_indoor_metadata(
        sots_root / "metadata_indoor.csv",
        indoor_assignments,
    )

    print("Đã chia dữ liệu theo tỷ lệ train:val:test = 5:1:1")
    print(f"O-HAZY: {o_hazy_counts}")
    print(f"SOTS indoor: {indoor_counts}")
    print(f"SOTS outdoor: {outdoor_counts}")


if __name__ == "__main__":
    main()

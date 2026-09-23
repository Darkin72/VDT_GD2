"""Create publication-style Excel summaries for dehazing evaluations."""

from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Border, Font, Side


DATASETS = (
    ("sots-indoor", "SOTS-Indoor"),
    ("sots-outdoor", "SOTS-Outdoor"),
    ("o-hazy", "O-HAZE"),
    ("i-haze", "I-HAZE"),
)
METRICS = ("PSNR", "SSIM", "ms / ảnh")


def summarize(rows):
    if not rows:
        return None
    return (
        float(np.mean([float(row["output_psnr"]) for row in rows])),
        float(np.mean([float(row["output_ssim"]) for row in rows])),
        float(np.mean([float(row["runtime_ms"]) for row in rows])),
    )


def style_sheet(sheet, last_column):
    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for row in sheet.iter_rows(min_row=1, max_row=4, min_col=1, max_col=last_column):
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in range(1, 4):
        for cell in sheet[row]:
            cell.font = Font(bold=True, size=14 if row == 1 else 11)
    sheet.column_dimensions["A"].width = 28
    for column in range(2, last_column + 1):
        sheet.column_dimensions[get_column_letter(column)].width = 14
    for row in range(1, 5):
        sheet.row_dimensions[row].height = 24
    sheet.freeze_panes = "B4"


def write_grouped_workbook(path, solution_name, groups):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Results"
    last_column = 1 + len(groups) * len(METRICS)

    sheet.merge_cells(start_row=1, start_column=1, end_row=3, end_column=1)
    sheet["A1"] = "Solution"
    sheet.merge_cells(start_row=1, start_column=2, end_row=1, end_column=last_column)
    sheet.cell(1, 2, "Dataset")

    column = 2
    for label, values in groups:
        sheet.merge_cells(start_row=2, start_column=column, end_row=2, end_column=column + 2)
        sheet.cell(2, column, label)
        for offset, metric in enumerate(METRICS):
            sheet.cell(3, column + offset, metric)
        if values:
            for offset, value in enumerate(values):
                cell = sheet.cell(4, column + offset, value)
                cell.number_format = "0.0000" if offset == 1 else "0.00"
        column += 3

    sheet.cell(4, 1, solution_name)
    sheet.cell(4, 1).alignment = Alignment(horizontal="left", vertical="center")
    style_sheet(sheet, last_column)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def export_excel_reports(solution_name, rows, output_dir):
    output_dir = Path(output_dir)
    slug = solution_name.lower().replace(" ", "_")

    dataset_groups = []
    for key, label in DATASETS:
        dataset_groups.append((label, summarize([row for row in rows if row["dataset"] == key])))

    origin_groups = []
    for key, label in (("real", "Real"), ("synthetic", "Synthetic")):
        origin_groups.append((label, summarize([row for row in rows if row["data_origin"] == key])))

    fog_groups = []
    for key, label in (("light", "Light fog"), ("medium", "Medium fog"), ("heavy", "Heavy fog")):
        fog_groups.append((label, summarize([row for row in rows if row["fog_level"] == key])))

    paths = {
        "dataset": output_dir / f"{slug}_dataset.xlsx",
        "domain": output_dir / f"{slug}_domain.xlsx",
        "fog": output_dir / f"{slug}_fog.xlsx",
    }
    write_grouped_workbook(paths["dataset"], solution_name, dataset_groups)
    write_grouped_workbook(paths["domain"], solution_name, origin_groups)
    write_grouped_workbook(paths["fog"], solution_name, fog_groups)
    return paths

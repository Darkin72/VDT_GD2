"""Hardware/performance workbook shared by model evaluators."""
import json
import os
import platform
import resource
import time
from pathlib import Path

import torch
from openpyxl import Workbook


def write_hardware_report(output_dir, solution, images, total_seconds, inference_seconds):
    output_dir = Path(output_dir)
    process_peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    hardware = {
        "RAM peak (GB)": round(process_peak_gb, 3),
        "FPS trung bình toàn bộ quá trình": images / total_seconds if total_seconds else 0,
        "FPS trung bình suy luận": images / inference_seconds if inference_seconds else 0,
        "VRAM peak (GB)": 0,
        "VRAM total (GB)": 0,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        hardware["VRAM peak (GB)"] = round(torch.cuda.max_memory_allocated(index) / 1024**3, 3)
        hardware["VRAM total (GB)"] = round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 3)
    performance = {"solution": solution, "images": images, "total_seconds": total_seconds, "inference_seconds": inference_seconds, "hardware": hardware}
    (output_dir / "performance.json").write_text(json.dumps(performance, indent=2, ensure_ascii=False), encoding="utf-8")
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Hardware"
    sheet.append(("Solution", solution))
    for key, value in hardware.items():
        sheet.append((key, value))
    sheet.append(("Images", images))
    sheet.append(("Total seconds", total_seconds))
    sheet.append(("Inference seconds", inference_seconds))
    for column in (1, 2):
        sheet.column_dimensions["AB"[column - 1]].width = 42 if column == 1 else 20
    workbook.save(output_dir / "hardware.xlsx")
    return performance

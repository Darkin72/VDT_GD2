"""Command line inference for HazeWaveNet."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from .model import HazeWaveNet

def main() -> None:
    ap = argparse.ArgumentParser(description="Dehaze one image with HazeWaveNet")
    ap.add_argument("input", type=Path); ap.add_argument("output", type=Path)
    ap.add_argument("--checkpoint", type=Path); ap.add_argument("--levels", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HazeWaveNet(levels=args.levels).to(device).eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
    image = Image.open(args.input).convert("RGB")
    tensor = torch.from_numpy(np.array(image, copy=True)).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        result = model(tensor)[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.output)

if __name__ == "__main__": main()

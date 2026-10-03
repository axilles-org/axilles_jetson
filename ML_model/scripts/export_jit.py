"""Export a trained run to a deployable module (TorchScript, optionally ONNX).

    python scripts/export_jit.py --run runs/<run_dir> --mass 72 --height 1.75 --gender M
    python scripts/export_jit.py --run runs/<run_dir> --mass 72 --onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.deploy.export import export


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mass", type=float, required=True, help="deployment subject mass (kg)")
    ap.add_argument("--height", type=float, default=None, help="subject height (m)")
    ap.add_argument("--gender", default=None, choices=["M", "F"])
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--onnx", action="store_true")
    args = ap.parse_args()

    demo = None
    if args.height is not None and args.gender is not None:
        demo = (args.height, args.mass, args.gender)

    artifacts = export(args.run, args.mass, subject_demo=demo,
                       onnx=args.onnx, checkpoint=args.checkpoint)
    print(f"torchscript : {artifacts.torchscript}")
    if artifacts.onnx:
        print(f"onnx        : {artifacts.onnx}")
    print(f"metadata    : {artifacts.metadata}")


if __name__ == "__main__":
    main()

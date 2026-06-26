"""Run FID + throughput comparison for Kuramoto and DCGAN checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kuramoto", type=Path, required=True)
    parser.add_argument("--dcgan", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/mnist_reals"))
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/comparison.json"))
    return parser


def _run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # fid/benchmark scripts print human output; load sidecar if written
    return json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    results: dict[str, dict] = {}

    for name, checkpoint, model in (
        ("kuramoto", args.kuramoto, "kuramoto"),
        ("dcgan", args.dcgan, "dcgan"),
    ):
        fid_out = Path("results") / f"{name}_fid.json"
        bench_out = Path("results") / f"{name}_bench.json"
        subprocess.run(
            [
                py,
                str(root / "eval" / "fid.py"),
                "--checkpoint",
                str(checkpoint),
                "--model",
                model,
                "--num-samples",
                str(args.num_samples),
                "--real-dir",
                str(args.real_dir),
                "--device",
                str(args.device),
                "--output",
                str(fid_out),
            ],
            check=True,
        )
        subprocess.run(
            [
                py,
                str(root / "eval" / "benchmark.py"),
                "--checkpoint",
                str(checkpoint),
                "--model",
                model,
                "--num-images",
                str(args.num_images),
                "--device",
                str(args.device),
                "--output",
                str(bench_out),
            ],
            check=True,
        )
        results[name] = {
            "fid": json.loads(fid_out.read_text()),
            "benchmark": json.loads(bench_out.read_text()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    print("\n| Model     | FID ↓ | ms/image | Params |")
    print("|-----------|-------|----------|--------|")
    for name in ("kuramoto", "dcgan"):
        fid = results[name]["fid"]["fid"]
        ms = results[name]["benchmark"]["ms_per_image"]
        params = results[name]["benchmark"]["params"]
        print(f"| {name:9s} | {fid:5.2f} | {ms:8.2f} | {params:,} |")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

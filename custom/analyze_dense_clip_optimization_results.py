#!/usr/bin/env python3
"""Summarize dense clip optimization benchmark results from TSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_RESULTS = Path(__file__).resolve().parent / "dense_clip_optimization_results.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dense clip optimization results TSV")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--profile", type=str, default="quality", help="Profile rows to summarize")
    return parser.parse_args()


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("inf")


def main() -> None:
    args = parse_args()
    if not args.results.exists():
        raise FileNotFoundError(f"Results TSV not found: {args.results}")

    with open(args.results, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    rows = [row for row in rows if row.get("profile") == args.profile]
    if not rows:
        print(f"No rows found for profile={args.profile}")
        return

    keep_rows = [row for row in rows if row.get("status") == "keep"]
    print(f"Total {args.profile} rows: {len(rows)}")
    print(f"Keep rows: {len(keep_rows)}")
    if rows:
        print(f"Latest row: {rows[-1]['commit']}  {rows[-1]['description']}")

    best_quality = min(rows, key=lambda row: as_float(row, "best_eval_mse"))
    best_speed = min(rows, key=lambda row: as_float(row, "projected_1000_s"))

    print()
    print("Best quality:")
    print(
        f"  commit={best_quality['commit']}  best_eval_mse={best_quality['best_eval_mse']}  "
        f"projected_1000_s={best_quality['projected_1000_s']}  status={best_quality['status']}  "
        f"{best_quality['description']}"
    )

    print()
    print("Best projected speed:")
    print(
        f"  commit={best_speed['commit']}  best_eval_mse={best_speed['best_eval_mse']}  "
        f"projected_1000_s={best_speed['projected_1000_s']}  status={best_speed['status']}  "
        f"{best_speed['description']}"
    )

    print()
    print("Kept frontier:")
    for row in keep_rows:
        print(
            f"  {row['commit']}  mse={row['best_eval_mse']}  total_s={row['total_s']}  "
            f"projected_1000_s={row['projected_1000_s']}  {row['description']}"
        )


if __name__ == "__main__":
    main()

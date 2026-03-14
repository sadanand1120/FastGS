#!/usr/bin/env python3
"""Summarize and plot dense CLIP optimization benchmark results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_RESULTS = Path(__file__).resolve().parent / "dense_clip_optimization_results.tsv"
DEFAULT_PLOT = Path(__file__).resolve().parent / "dense_clip_optimization_progress.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dense clip optimization results TSV")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--profile", type=str, default="quality", help="Profile rows to summarize")
    parser.add_argument("--plot-out", type=Path, default=DEFAULT_PLOT)
    return parser.parse_args()


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("inf")


def load_rows(results: Path, profile: str) -> list[dict[str, object]]:
    with open(results, "r", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f, delimiter="\t"))

    rows: list[dict[str, object]] = []
    experiment = 0
    for row in raw_rows:
        if row.get("profile") != profile:
            continue
        experiment += 1
        parsed = dict(row)
        parsed["experiment"] = experiment
        parsed["best_eval_mse_value"] = as_float(row, "best_eval_mse")
        parsed["projected_1000_s_value"] = as_float(row, "projected_1000_s")
        parsed["total_s_value"] = as_float(row, "total_s")
        parsed["status_norm"] = row.get("status", "").strip().upper()
        rows.append(parsed)
    return rows


def summarize(rows: list[dict[str, object]], profile: str) -> None:
    keep_rows = [row for row in rows if row["status_norm"] == "KEEP"]
    print(f"Total {profile} rows: {len(rows)}")
    print(f"Keep rows: {len(keep_rows)}")
    if rows:
        print(f"Latest row: {rows[-1]['commit']}  {rows[-1]['description']}")

    best_quality = min(rows, key=lambda row: row["best_eval_mse_value"])
    best_speed = min(rows, key=lambda row: row["projected_1000_s_value"])

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


def save_progress_plot(rows: list[dict[str, object]], out_path: Path, profile: str) -> None:
    kept_rows = [row for row in rows if row["status_norm"] == "KEEP"]
    discarded_rows = [row for row in rows if row["status_norm"] == "DISCARD"]
    crash_rows = [row for row in rows if row["status_norm"] == "CRASH"]

    baseline_runtime = float(rows[0]["projected_1000_s_value"])

    running_best_x: list[int] = []
    running_best_y: list[float] = []
    best_so_far = float("inf")
    for row in kept_rows:
        value = float(row["projected_1000_s_value"])
        best_so_far = min(best_so_far, value)
        running_best_x.append(int(row["experiment"]))
        running_best_y.append(best_so_far)

    fig, ax = plt.subplots(figsize=(16, 8))

    if discarded_rows:
        ax.scatter(
            [row["experiment"] for row in discarded_rows],
            [row["projected_1000_s_value"] for row in discarded_rows],
            alpha=0.4,
            color="#c7c7c7",
            s=30,
            label="Discarded",
            zorder=2,
        )

    if crash_rows:
        ax.scatter(
            [row["experiment"] for row in crash_rows],
            [row["projected_1000_s_value"] for row in crash_rows],
            alpha=0.8,
            color="#e74c3c",
            s=40,
            marker="x",
            label="Crash",
            zorder=3,
        )

    if kept_rows:
        ax.scatter(
            [row["experiment"] for row in kept_rows],
            [row["projected_1000_s_value"] for row in kept_rows],
            alpha=0.95,
            color="#2e8b57",
            edgecolors="black",
            linewidth=0.5,
            s=60,
            label="Kept",
            zorder=4,
        )
        ax.step(
            running_best_x,
            running_best_y,
            where="post",
            color="#1b5e20",
            linewidth=2.5,
            alpha=0.95,
            label="Best so far",
            zorder=3,
        )

        y_min = min(float(row["projected_1000_s_value"]) for row in rows)
        y_max = max(float(row["projected_1000_s_value"]) for row in rows)
        y_offset = max((y_max - y_min) * 0.015, baseline_runtime * 0.005)
        for row in kept_rows:
            ax.annotate(
                str(row["description"]),
                (int(row["experiment"]), float(row["projected_1000_s_value"])),
                xytext=(5, y_offset),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
                rotation=28,
                ha="left",
                color="#1b5e20",
            )

    ax.set_xlabel("Experiment Number")
    ax.set_ylabel("Projected Runtime for 1000 Images (s)")
    ax.set_title(
        f"Dense CLIP Optimization Progress: {len(rows)} {profile} Experiments, {len(kept_rows)} Kept"
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    ax.axhline(
        baseline_runtime,
        color="#4f6d7a",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
    )
    summary_lines = [
        f"Baseline: {baseline_runtime:.1f}s",
        f"Best quality: {min(rows, key=lambda row: row['best_eval_mse_value'])['best_eval_mse']}",
    ]
    if kept_rows:
        best_kept_speed = min(kept_rows, key=lambda row: row["projected_1000_s_value"])
        summary_lines.insert(1, f"Best kept speed: {best_kept_speed['projected_1000_s']}s")
    ax.text(
        0.01,
        0.02,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.9},
        va="bottom",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> None:
    args = parse_args()
    if not args.results.exists():
        raise FileNotFoundError(f"Results TSV not found: {args.results}")

    rows = load_rows(args.results, args.profile)
    if not rows:
        print(f"No rows found for profile={args.profile}")
        return

    summarize(rows, args.profile)
    save_progress_plot(rows, args.plot_out.resolve(), args.profile)


if __name__ == "__main__":
    main()

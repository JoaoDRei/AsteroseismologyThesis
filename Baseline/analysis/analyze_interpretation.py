#!/usr/bin/env python3
"""Summarize interpretation run status and generated saliency maps.

Run from the thesis project root:
    python -m Baseline.analysis.analyze_interpretation

Writes interpretation run/target-coverage tables and a map-coverage plot to
Baseline/results_analysis. Saliency images remain in their experiment folders.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from Baseline.paths import BASELINE_DIR


ROOT = BASELINE_DIR.parent
DEFAULT_PLAN = BASELINE_DIR / "interpretation" / "interpretation_plan.csv"
DEFAULT_STATUS_DIR = BASELINE_DIR / "interpretation" / "status"
DEFAULT_OUTPUT_DIR = BASELINE_DIR / "results_analysis"


def read_json(path):
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def targets_from_row(row):
    return [
        target.strip()
        for target in (row.get("targets") or "").split("+")
        if target.strip()
    ]


def collect_run_record(row, status_dir):
    analysis_id = row["analysis_id"]
    status = read_json(status_dir / f"{analysis_id}.json")
    run_dir = resolve_path(row["run_dir"])
    output_dir = Path(status.get("output_dir") or run_dir / "interpretability")
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    method = (row.get("method") or "").strip().lower()
    targets = targets_from_row(row)
    map_paths = []
    kics = set()
    target_records = []
    for target in targets:
        suffix = f"_{target}_{method}.png"
        target_paths = [
            path for path in output_dir.glob(f"kic_*{suffix}")
            if path.is_file()
        ] if output_dir.is_dir() else []
        target_kics = {
            path.stem[len("kic_"):-len(suffix)]
            for path in target_paths
            if path.stem.startswith("kic_")
        }
        map_paths.extend(target_paths)
        kics.update(target_kics)
        target_records.append({
            "analysis_id": analysis_id,
            "target": target,
            "method": method,
            "map_count": len(target_paths),
            "unique_kic_count": len(target_kics),
            "max_samples": row.get("max_samples"),
            "output_dir": str(output_dir),
        })

    return {
        "analysis_id": analysis_id,
        "model": row.get("model"),
        "input": row.get("mode"),
        "duration": row.get("duration"),
        "targets": "+".join(targets),
        "method": method,
        "status": status.get("status", "not_started"),
        "return_code": status.get("return_code"),
        "complete": (
            status.get("status") == "completed"
            and status.get("return_code") == 0
            and bool(map_paths)
        ),
        "map_count": len(map_paths),
        "unique_kic_count": len(kics),
        "max_samples": row.get("max_samples"),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "reason": status.get("reason"),
    }, target_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--status-dir", type=Path, default=DEFAULT_STATUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.plan.is_file():
        raise FileNotFoundError(f"Interpretation plan not found: {args.plan}")
    output_dir = args.output_dir.expanduser().resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with args.plan.open("r", newline="", encoding="utf-8") as plan_file:
        jobs = list(csv.DictReader(plan_file))

    records = []
    target_records = []
    for row in jobs:
        record, per_target = collect_run_record(row, args.status_dir)
        records.append(record)
        target_records.extend(per_target)

    runs_df = pd.DataFrame(records)
    runs_path = output_dir / "interpretation_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    targets_path = output_dir / "interpretation_target_coverage.csv"
    pd.DataFrame(target_records).to_csv(targets_path, index=False)

    if not runs_df.empty and target_records:
        coverage_df = pd.DataFrame(target_records)
        coverage_df["max_samples"] = pd.to_numeric(
            coverage_df["max_samples"], errors="coerce"
        )
        coverage_df["coverage"] = (
            coverage_df["map_count"] / coverage_df["max_samples"]
        )
        analysis_order = runs_df["analysis_id"].tolist()
        target_order = coverage_df["target"].drop_duplicates().tolist()
        coverage = coverage_df.pivot(
            index="analysis_id", columns="target", values="coverage"
        ).reindex(index=analysis_order, columns=target_order)
        counts = coverage_df.pivot(
            index="analysis_id", columns="target", values="map_count"
        ).reindex(index=analysis_order, columns=target_order)
        caps = coverage_df.pivot(
            index="analysis_id", columns="target", values="max_samples"
        ).reindex(index=analysis_order, columns=target_order)

        fig, ax = plt.subplots(
            figsize=(max(8, len(target_order) * 2.2),
                     max(4.5, len(analysis_order) * 0.65))
        )
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#e5e7eb")
        image = ax.imshow(coverage.to_numpy(dtype=float), cmap=cmap,
                          vmin=0, vmax=1, aspect="auto")
        for row_index, analysis_id in enumerate(analysis_order):
            for column_index, target in enumerate(target_order):
                count = counts.loc[analysis_id, target]
                cap = caps.loc[analysis_id, target]
                if pd.isna(count):
                    continue
                if pd.notna(cap) and cap > 0:
                    label = f"{int(count)} / {int(cap)}"
                    value = coverage.loc[analysis_id, target]
                    text_color = "white" if value >= 0.55 else "#17212b"
                else:
                    label = f"{int(count)}"
                    text_color = "#17212b"
                ax.text(column_index, row_index, label, ha="center",
                        va="center", color=text_color, fontsize=9)

        status_by_analysis = runs_df.set_index("analysis_id")["status"]
        ax.set_yticks(range(len(analysis_order)))
        ax.set_yticklabels([
            f"{analysis_id}  ({status_by_analysis[analysis_id]})"
            for analysis_id in analysis_order
        ])
        ax.set_xticks(range(len(target_order)))
        ax.set_xticklabels(target_order)
        ax.set_xlabel("Prediction target")
        ax.set_ylabel("Analysis (run status)")
        ax.set_title("Saliency-map coverage by target")
        colorbar = fig.colorbar(image, ax=ax, pad=0.02, extend="max")
        colorbar.set_label("Fraction of requested map cap")
        colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        colorbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xticks(
            [position - 0.5 for position in range(1, len(target_order))],
            minor=True,
        )
        ax.set_yticks(
            [position - 0.5 for position in range(1, len(analysis_order))],
            minor=True,
        )
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.tight_layout()
        plot_path = plots_dir / "interpretation_map_coverage.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(f"Saved {plot_path}")

    print(f"Saved {runs_path}")
    print(f"Saved {targets_path}")
    print(
        f"Completed analyses with map outputs: "
        f"{int(runs_df['complete'].sum())}/{len(runs_df)}"
    )


if __name__ == "__main__":
    main()

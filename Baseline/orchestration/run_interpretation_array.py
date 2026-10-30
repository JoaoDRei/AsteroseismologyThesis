
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from Baseline.paths import BASELINE_DIR, PROJECT_ROOT

ROOT = PROJECT_ROOT
PLAN = BASELINE_DIR / "interpretation" / "interpretation_plan.csv"
INTERPRET_MODULE = "Baseline.training.interpret"
SLURM_LOGS = BASELINE_DIR / "slurm_logs" / "interpretation"
SKIP_MESSAGE = "Already completed; skipping."


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def archive_current_slurm_logs(task_id):
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    if array_job_id is None:
        return

    completed_dir = SLURM_LOGS / "completed"
    for extension in ("out", "err"):
        log_path = SLURM_LOGS / f"interpret_{array_job_id}_{task_id}.{extension}"
        if not log_path.is_file():
            print(
                f"WARNING: Slurm log not found; could not archive {log_path}",
                file=sys.stderr,
                flush=True,
            )
            continue

        try:
            completed_dir.mkdir(parents=True, exist_ok=True)
            archived_path = completed_dir / log_path.name
            if archived_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite archived Slurm log: {archived_path}"
                )
            log_path.rename(archived_path)
        except OSError as exc:
            print(
                f"WARNING: Could not archive Slurm log {log_path}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def archive_previous_completed_logs(task_id):
    current_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    completed_dir = SLURM_LOGS / "completed"

    for output_path in SLURM_LOGS.glob("interpret_*.out"):
        log_id = output_path.stem.removeprefix("interpret_")
        array_job_id, separator, log_task_id = log_id.partition("_")
        if not separator or (
            array_job_id == current_array_job_id and log_task_id == task_id
        ):
            continue

        try:
            with output_path.open(errors="replace") as log_file:
                is_completed = any(
                    "COMPLETED:" in line or SKIP_MESSAGE in line
                    for line in log_file
                )
            if not is_completed:
                continue

            completed_dir.mkdir(parents=True, exist_ok=True)
            for extension in ("out", "err"):
                source = SLURM_LOGS / f"{output_path.stem}.{extension}"
                if not source.is_file():
                    continue

                archived_path = completed_dir / source.name
                if archived_path.exists():
                    print(
                        f"WARNING: Archived Slurm log already exists; "
                        f"leaving {source} in place",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                try:
                    source.rename(archived_path)
                except FileNotFoundError:
                    # Another array task may have archived this file first.
                    continue
        except FileNotFoundError:
            # Another array task may have moved this output log.
            continue
        except OSError as exc:
            print(
                f"WARNING: Could not move completed interpretation logs for "
                f"{output_path.stem} into {completed_dir}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def main():
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is missing. "
            "Submit through the SLURM array."
        )

    archive_previous_completed_logs(task_id)

    if not PLAN.is_file():
        raise FileNotFoundError(f"Interpretation plan not found: {PLAN}")

    with PLAN.open("r", newline="") as f:
        jobs = list(csv.DictReader(f))

    index = int(task_id)
    if not 0 <= index < len(jobs):
        raise IndexError(
            f"Task index {index} is invalid; "
            f"plan contains {len(jobs)} jobs."
        )

    job = jobs[index]
    analysis_id = job["analysis_id"]
    run_dir = resolve_path(job["run_dir"])
    output_dir = run_dir / "interpretability"
    status_path = (
        ROOT / "Baseline/interpretation/status" /
        f"{analysis_id}.json"
    )

    print(f"Analysis {index + 1}/{len(jobs)}: {analysis_id}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Method: {job.get('method', 'config')}", flush=True)

    if not run_dir.is_dir():
        write_status(status_path, {
            "analysis_id": analysis_id,
            "status": "failed",
            "reason": f"Training run directory not found: {run_dir}",
            "finished_at": now(),
        })
        raise FileNotFoundError(run_dir)

    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "best_model.pth"

    missing = [
        str(p) for p in
        (config_path, checkpoint_path)
        if not p.is_file()
    ]
    if missing:
        write_status(status_path, {
            "analysis_id": analysis_id,
            "status": "failed",
            "reason": f"Required run files missing: {missing}",
            "finished_at": now(),
        })
        raise FileNotFoundError(", ".join(missing))

    # Do not skip a previous failure, but avoid duplicate completed jobs.
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("status") == "completed":
            print("Already completed; skipping.", flush=True)
            archive_current_slurm_logs(index)
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "analysis_id": analysis_id,
        "status": "running",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "started_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": index,
    }
    write_status(status_path, status)

    command = [
        sys.executable,
        "-m",
        INTERPRET_MODULE,
        "--run-dir", str(run_dir),
        "--output-dir", str(output_dir),
    ]

    method = (job.get("method") or "").strip().lower()
    if method:
        command.extend(["--method", method])

    max_samples = (job.get("max_samples") or "").strip()
    if max_samples:
        command.extend(["--max-samples", max_samples])

    kics = (job.get("kics") or "").strip()
    if kics:
        command.extend(["--kics", kics])

    print("Command:", " ".join(command), flush=True)

    try:
        result = subprocess.run(command, cwd=ROOT, check=False)

        status["return_code"] = result.returncode
        status["finished_at"] = now()
        status["status"] = (
            "completed" if result.returncode == 0 else "failed"
        )

        if result.returncode != 0:
            status["reason"] = (
                "interpret.py exited with a non-zero return code"
            )

        write_status(status_path, status)

        if result.returncode != 0:
            sys.exit(result.returncode)

        print(f"COMPLETED: {analysis_id}", flush=True)
        archive_current_slurm_logs(index)

    except Exception as exc:
        status["status"] = "failed"
        status["reason"] = repr(exc)
        status["finished_at"] = now()
        write_status(status_path, status)
        raise


if __name__ == "__main__":
    main()

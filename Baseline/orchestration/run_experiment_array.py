
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from Baseline.paths import BASELINE_DIR, PROJECT_ROOT

ROOT = PROJECT_ROOT
PLAN = BASELINE_DIR / "experiments" / "experiment_plan.csv"
TRAIN_MODULES = {
    "train": "Baseline.training.train",
    "cross_validate": "Baseline.training.cross_validate",
    "autoencoder": "Baseline.training.train_autoencoder",
}
SLURM_LOGS = BASELINE_DIR / "slurm_logs"
SKIP_MESSAGE = "Already completed; skipping."


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def archive_slurm_logs(task_id):
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    if array_job_id is None:
        return

    completed_dir = SLURM_LOGS / "completed"
    for extension in ("out", "err"):
        log_path = SLURM_LOGS / f"{array_job_id}_{task_id}.{extension}"
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
            shutil.copy2(log_path, archived_path)
        except OSError as exc:
            print(
                f"WARNING: Could not archive Slurm log {log_path}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def archive_previous_skipped_logs():
    current_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    completed_dir = SLURM_LOGS / "completed"
    log_stems = {
        log_path.stem
        for pattern in ("*.out", "*.err")
        for log_path in SLURM_LOGS.glob(pattern)
    }

    for log_stem in log_stems:
        array_job_id, separator, _ = log_stem.partition("_")
        if not separator or array_job_id == current_array_job_id:
            continue

        try:
            log_path = SLURM_LOGS / f"{log_stem}.out"
            archived_output = completed_dir / log_path.name
            output_to_check = (
                log_path if log_path.is_file() else archived_output
            )
            if not output_to_check.is_file():
                continue

            is_experiment_log = False
            is_skipped_run = False
            with output_to_check.open(errors="replace") as log_file:
                for line in log_file:
                    is_experiment_log |= line.startswith("Experiment ")
                    is_skipped_run |= SKIP_MESSAGE in line
                    if is_experiment_log and is_skipped_run:
                        break
            if not is_experiment_log or not is_skipped_run:
                continue

            completed_dir.mkdir(parents=True, exist_ok=True)
            for extension in ("out", "err"):
                source = SLURM_LOGS / f"{log_stem}.{extension}"
                if not source.is_file():
                    continue

                archived_path = completed_dir / source.name
                if not archived_path.exists():
                    try:
                        source.rename(archived_path)
                    except FileNotFoundError:
                        # Another array task may have archived this file first.
                        continue
                    continue

                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=completed_dir,
                        prefix=f".{source.name}.",
                        delete=False,
                    ) as temp_file:
                        temp_path = Path(temp_file.name)
                    shutil.copy2(source, temp_path)
                    temp_path.replace(archived_path)
                    source.unlink()
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
        except FileNotFoundError:
            # Another array task may have moved a matching log concurrently.
            continue
        except OSError as exc:
            print(
                f"WARNING: Could not move completed skip logs for "
                f"{log_path.stem} into {completed_dir}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def main():
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if task_id is None:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is missing. "
            "Submit this script through the SLURM array."
        )

    archive_previous_skipped_logs()

    index = int(task_id)

    with PLAN.open("r", newline="") as f:
        experiments = list(csv.DictReader(f))

    if not 0 <= index < len(experiments):
        raise IndexError(
            f"Task index {index} is invalid; "
            f"experiment plan contains {len(experiments)} rows."
        )

    exp = experiments[index]
    name = exp["experiment"]
    trainer = exp.get("trainer", "train") or "train"
    if trainer not in TRAIN_MODULES:
        raise ValueError(f"Unsupported trainer {trainer!r} for experiment {name}")

    config = ROOT / exp["config_path"]
    exp_dir = config.parent
    status_path = exp_dir / "execution_status.json"

    print(f"Experiment {index + 1}/{len(experiments)}: {name}", flush=True)
    print(f"Trainer: {trainer}", flush=True)
    print(f"Model: {exp['model']}", flush=True)
    print(f"Input: {exp['input']}", flush=True)
    print(f"Duration: {exp['duration']}", flush=True)
    print(f"Targets: {exp['targets']}", flush=True)
    print(f"Config: {config}", flush=True)

    if not config.is_file():
        write_status(status_path, {
            "experiment": name,
            "status": "failed",
            "reason": "Configuration file not found",
            "config_path": str(config),
            "finished_at": now(),
        })
        raise FileNotFoundError(config)

    # Avoid accidentally rerunning experiments that already completed.
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("status") == "completed":
            print(SKIP_MESSAGE, flush=True)
            archive_slurm_logs(index)
            return

    status = {
        "experiment": name,
        "status": "running",
        "config_path": str(config),
        "started_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": index,
    }
    write_status(status_path, status)

    command = [
        sys.executable,
        "-m",
        TRAIN_MODULES[trainer],
        "--config",
        str(config),
    ]

    print("Command:", " ".join(command), flush=True)

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
        )

        status["return_code"] = result.returncode
        status["finished_at"] = now()
        status["status"] = (
            "completed" if result.returncode == 0 else "failed"
        )

        if result.returncode != 0:
            status["reason"] = (
                "Training process exited with a non-zero return code"
            )

        write_status(status_path, status)

        if result.returncode != 0:
            print(
                f"FAILED: {name}, exit code {result.returncode}",
                flush=True,
            )
            sys.exit(result.returncode)

        print(f"COMPLETED: {name}", flush=True)

    except Exception as exc:
        status["status"] = "failed"
        status["reason"] = repr(exc)
        status["finished_at"] = now()
        write_status(status_path, status)
        raise


if __name__ == "__main__":
    main()

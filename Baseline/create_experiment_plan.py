
from pathlib import Path
import copy
import csv
import yaml

ROOT = Path("./Baseline")
BASE_CONFIG = ROOT / "config.yaml"
EXPERIMENTS_DIR = ROOT / "experiments"

# Targets used for the first architecture comparison.
MAIN_TARGETS = ["nu_max", "delta_nu", "class"]

experiments = [
    # Phase A: architecture comparison
    ("01_architecture", "A0_cnn_lc_80d", "cnn", "lc", "80d", MAIN_TARGETS, "Existing baseline; verify config and results."),
    ("01_architecture", "A1_resnet1d_lc_80d", "resnet1d", "lc", "80d", MAIN_TARGETS, "Residual convolutional baseline."),
    ("01_architecture", "A2_mobilenet1d_lc_80d", "mobilenet1d", "lc", "80d", MAIN_TARGETS, "Efficient convolutional model."),
    ("01_architecture", "A3_cnn_lstm_lc_80d", "cnn_lstm", "lc", "80d", MAIN_TARGETS, "CNN features followed by an LSTM."),
    ("01_architecture", "A4_cnn_bilstm_lc_80d", "cnn_bilstm", "lc", "80d", MAIN_TARGETS, "CNN features followed by a BiLSTM."),
    ("01_architecture", "A5_astroconformer_lc_80d", "astroconformer", "lc", "80d", MAIN_TARGETS, "Convolutional patch embedding."),
    ("01_architecture", "A6_astroconformer_v2_lc_80d", "astroconformer_v2", "lc", "80d", MAIN_TARGETS, "Dense patch embedding ablation."),

    # Phase B: representation comparison; choose final models later.
    ("02_representation", "B0_cnn_psd_80d", "cnn", "psd", "80d", MAIN_TARGETS, "PSD comparison against the existing LC CNN."),
    ("02_representation", "B1_resnet1d_psd_80d", "resnet1d", "psd", "80d", MAIN_TARGETS, "PSD comparison with residual convolutions."),

    # Phase C: duration comparison; select the architecture after Phase A.
    ("03_duration", "C0_selected_model_lc_20d", "resnet1d", "lc", "20d", MAIN_TARGETS, "Change model to the Phase A winner."),
    ("03_duration", "C1_selected_model_lc_55d", "resnet1d", "lc", "55d", MAIN_TARGETS, "Change model to the Phase A winner."),
    ("03_duration", "C2_selected_model_lc_80d", "resnet1d", "lc", "80d", MAIN_TARGETS, "Change model to the Phase A winner."),

    # Phase D: exploratory metallicity experiments.
    ("04_metallicity", "D0_cnn_metallicity_lc_80d", "cnn", "lc", "80d", ["metallicity"], "Metallicity-only baseline."),
    ("04_metallicity", "D1_selected_model_metallicity_lc_80d", "resnet1d", "lc", "80d", ["metallicity"], "Change model after Phase A."),
]

if not BASE_CONFIG.exists():
    raise FileNotFoundError(
        f"Could not find {BASE_CONFIG}. Run this script from the project root."
    )

with open(BASE_CONFIG, "r") as f:
    base_config = yaml.safe_load(f)

plan_rows = []

for phase, name, model, mode, duration, targets, purpose in experiments:
    exp_dir = EXPERIMENTS_DIR / phase / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(base_config)
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})

    cfg["model"]["type"] = model
    cfg["model"]["targets"] = targets
    cfg["data"]["mode"] = mode
    cfg["data"]["filter_duration"] = duration

    # Keep target-specific settings consistent with the selected targets.
    # If your base config has these dictionaries, retain only relevant keys.
    for key in ("loss_functions", "loss_weights"):
        if key in cfg["model"]:
            missing = [t for t in targets if t not in cfg["model"][key]]
            if missing:
                print(f"WARNING {name}: add {key} for targets {missing}")
            cfg["model"][key] = {
                t: cfg["model"][key][t]
                for t in targets
                if t in cfg["model"][key]
            }

    config_path = exp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    readme_path = exp_dir / "README.md"
    readme_path.write_text(
        f"# {name}\n\n"
        f"- **Phase:** {phase}\n"
        f"- **Model:** {model}\n"
        f"- **Input:** {mode}\n"
        f"- **Duration:** {duration}\n"
        f"- **Targets:** {', '.join(targets)}\n"
        f"- **Purpose:** {purpose}\n\n"
        "## Before running\n"
        "- Verify loss functions and target weights in config.yaml.\n"
        "- Verify the manifest and input paths.\n"
        "- Use the same KIC-level split as the other comparison runs.\n"
        "- Record any deviations from the planned experiment.\n\n"
        "## Results\n"
        "- Run directory:\n"
        "- Best epoch:\n"
        "- Test metrics:\n"
        "- Notes:\n"
    )

    plan_rows.append({
        "phase": phase,
        "experiment": name,
        "model": model,
        "input": mode,
        "duration": duration,
        "targets": "+".join(targets),
        "config_path": str(config_path),
        "status": "planned",
        "purpose": purpose,
    })

plan_path = EXPERIMENTS_DIR / "experiment_plan.csv"
plan_path.parent.mkdir(parents=True, exist_ok=True)

with open(plan_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=plan_rows[0].keys())
    writer.writeheader()
    writer.writerows(plan_rows)

print(f"Created {len(experiments)} experiment folders.")
print(f"Experiment plan: {plan_path}")
print("Review each config before submitting jobs.")
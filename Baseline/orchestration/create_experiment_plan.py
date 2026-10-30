
from pathlib import Path
import copy
import csv
import yaml
from Baseline.paths import BASELINE_DIR, PROJECT_ROOT

ROOT = BASELINE_DIR
BASE_CONFIG = ROOT / "config.yaml"
EXPERIMENTS_DIR = ROOT / "experiments"

# =========================================================
# Targets
# =========================================================

MAIN_TARGETS = ["nu_max", "delta_nu", "class"]
ASTERO_TARGETS = ["nu_max", "delta_nu"]

# =========================================================
# Phases 01-06: existing experiments
# =========================================================

experiments = [
    # Phase 01: architecture comparison (LC)
    (
        "01_architecture", "A0_cnn_lc_80d", "cnn", "lc", "80d",
        MAIN_TARGETS, "Existing baseline; verify config and results."
    ),
    (
        "01_architecture", "A1_resnet1d_lc_80d", "resnet1d", "lc", "80d",
        MAIN_TARGETS, "Residual convolutional baseline."
    ),
    (
        "01_architecture", "A2_mobilenet1d_lc_80d", "mobilenet1d", "lc", "80d",
        MAIN_TARGETS, "Efficient convolutional model."
    ),
    (
        "01_architecture", "A3_cnn_lstm_lc_80d", "cnn_lstm", "lc", "80d",
        MAIN_TARGETS, "CNN features followed by an LSTM."
    ),
    (
        "01_architecture", "A4_cnn_bilstm_lc_80d", "cnn_bilstm", "lc", "80d",
        MAIN_TARGETS, "CNN features followed by a BiLSTM."
    ),
    (
        "01_architecture", "A5_astroconformer_lc_80d", "astroconformer", "lc", "80d",
        MAIN_TARGETS, "Convolutional patch embedding."
    ),
    (
        "01_architecture", "A6_astroconformer_v2_lc_80d", "astroconformer_v2", "lc", "80d",
        MAIN_TARGETS, "Dense patch embedding ablation."
    ),

    # Phase 02: representation comparison (PSD)
    (
        "02_representation", "B0_cnn_psd_80d", "cnn", "psd", "80d",
        MAIN_TARGETS, "PSD comparison against the LC CNN."
    ),
    (
        "02_representation", "B1_resnet1d_psd_80d", "resnet1d", "psd", "80d",
        MAIN_TARGETS, "PSD comparison with residual convolutions."
    ),
    (
        "02_representation", "B2_astroconformer_psd_80d", "astroconformer", "psd", "80d",
        MAIN_TARGETS, "AstroConformer: PSD versus LC input."
    ),
    (
        "02_representation", "B3_astroconformer_v2_psd_80d", "astroconformer_v2", "psd", "80d",
        MAIN_TARGETS, "Dense-patch transformer: PSD versus LC input."
    ),

    # Phase 03: duration comparison
    (
        "03_duration", "C0_selected_model_lc_20d", "resnet1d", "lc", "20d",
        MAIN_TARGETS, "Change model to the Phase 01 winner."
    ),
    (
        "03_duration", "C1_selected_model_lc_55d", "resnet1d", "lc", "55d",
        MAIN_TARGETS, "Change model to the Phase 01 winner."
    ),
    (
        "03_duration", "C2_selected_model_lc_80d", "resnet1d", "lc", "80d",
        MAIN_TARGETS, "Change model to the Phase 01 winner."
    ),

    # Phase 04: exploratory metallicity experiments
    (
        "04_metallicity", "D0_cnn_metallicity_lc_80d", "cnn", "lc", "80d",
        ["metallicity"], "Metallicity-only baseline."
    ),
    (
        "04_metallicity", "D1_selected_model_metallicity_lc_80d", "resnet1d", "lc", "80d",
        ["metallicity"], "Change model after Phase 01."
    ),

    # Phase 05: regression-only experiments
    # These remove the evolutionary-classification output.
    (
        "05_two_target", "E0_cnn_lc_80d_2targets", "cnn", "lc", "80d",
        ASTERO_TARGETS, "CNN; regression-only multitask baseline."
    ),
    (
        "05_two_target", "E1_resnet1d_lc_80d_2targets", "resnet1d", "lc", "80d",
        ASTERO_TARGETS, "ResNet1D; regression-only multitask baseline."
    ),
    (
        "05_two_target", "E2_mobilenet1d_lc_80d_2targets", "mobilenet1d", "lc", "80d",
        ASTERO_TARGETS, "MobileNet1D; regression-only multitask baseline."
    ),
    (
        "05_two_target", "E3_cnn_lstm_lc_80d_2targets", "cnn_lstm", "lc", "80d",
        ASTERO_TARGETS, "CNN-LSTM; regression-only multitask baseline."
    ),
    (
        "05_two_target", "E4_cnn_bilstm_lc_80d_2targets", "cnn_bilstm", "lc", "80d",
        ASTERO_TARGETS, "CNN-BiLSTM; regression-only multitask baseline."
    ),
    (
        "05_two_target", "E5_astroconformer_lc_80d_2targets", "astroconformer", "lc", "80d",
        ASTERO_TARGETS, "AstroConformer; regression-only multitask baseline."
    ),
    (
        "05_two_target", "E6_astroconformer_v2_lc_80d_2targets", "astroconformer_v2", "lc", "80d",
        ASTERO_TARGETS, "AstroConformer v2; regression-only multitask baseline."
    ),
    (
        "05_two_target", "E7_cnn_psd_80d_2targets", "cnn", "psd", "80d",
        ASTERO_TARGETS, "CNN; PSD regression-only baseline."
    ),
    (
        "05_two_target", "E8_resnet1d_psd_80d_2targets", "resnet1d", "psd", "80d",
        ASTERO_TARGETS, "ResNet1D; PSD regression-only baseline."
    ),
    (
        "05_two_target", "E9_astroconformer_psd_80d_2targets", "astroconformer", "psd", "80d",
        ASTERO_TARGETS, "AstroConformer; PSD regression-only baseline."
    ),
    (
        "05_two_target", "E10_astroconformer_v2_psd_80d_2targets", "astroconformer_v2", "psd", "80d",
        ASTERO_TARGETS, "AstroConformer v2; PSD regression-only baseline."
    ),

    # Phase 06: unsupervised representation learning
    (
        "06_autoencoder",
        "F0_cnn_bilstm_autoencoder_lc_80d",
        "cnn_bilstm_autoencoder",
        "lc",
        "80d",
        [],
        "Unsupervised CNN-BiLSTM reconstruction baseline.",
        "autoencoder",
    ),
]

# =========================================================
# Phase 07: cross-validation experiments
# These are training/evaluation jobs handled by
# cross_validate.py, NOT interpretability jobs.
# =========================================================

CV_TARGETS = ["nu_max", "delta_nu", "class"]
CV_ASTERO_TARGETS = ["nu_max", "delta_nu"]

N_FOLDS = 5
CV_SEED = 42

cv_experiments = [
    (
        "07_cross_validation",
        "CV0_resnet1d_lc_80d",
        "resnet1d",
        "lc",
        "80d",
        CV_TARGETS,
        "Five-fold KIC-level CV for the ResNet1D multitask baseline.",
        "cross_validate",
    ),
    (
        "07_cross_validation",
        "CV1_cnn_lc_80d",
        "cnn",
        "lc",
        "80d",
        CV_TARGETS,
        "Five-fold KIC-level CV for the CNN multitask baseline.",
        "cross_validate",
    ),
    (
        "07_cross_validation",
        "CV2_cnn_lstm_lc_80d",
        "cnn_lstm",
        "lc",
        "80d",
        CV_TARGETS,
        "Five-fold KIC-level CV for the CNN-LSTM multitask baseline.",
        "cross_validate",
    ),
    (
        "07_cross_validation",
        "CV3_astroconformer_v2_psd_80d",
        "astroconformer_v2",
        "psd",
        "80d",
        CV_TARGETS,
        "Five-fold KIC-level CV for the PSD AstroConformer v2.",
        "cross_validate",
    ),
    (
        "07_cross_validation",
        "CV4_resnet1d_lc_80d_2targets",
        "resnet1d",
        "lc",
        "80d",
        CV_ASTERO_TARGETS,
        "Five-fold KIC-level CV for regression-only ResNet1D.",
        "cross_validate",
    ),
]

phase_order = {
    "01_architecture": 1,
    "02_representation": 2,
    "05_two_target": 3,
    "03_duration": 4,
    "04_metallicity": 5,
    "06_autoencoder": 6,
    "07_cross_validation": 7,
}
all_experiments = sorted(
    experiments,
    key=lambda experiment: phase_order[experiment[0]],
) + cv_experiments

# =========================================================
# Generate experiment configs, READMEs, and plan
# =========================================================

if not BASE_CONFIG.exists():
    raise FileNotFoundError(
        f"Could not find {BASE_CONFIG}. Run this script from the project root."
    )

with open(BASE_CONFIG, "r") as f:
    base_config = yaml.safe_load(f)

plan_rows = []

for experiment in all_experiments:
    phase, name, model, mode, duration, targets, purpose = experiment[:7]
    trainer = experiment[7] if len(experiment) > 7 else "train"

    exp_dir = EXPERIMENTS_DIR / phase / name
    config_path = exp_dir / "config.yaml"
    readme_path = exp_dir / "README.md"

    plan_rows.append({
        "phase": phase,
        "experiment": name,
        "trainer": trainer,
        "model": model,
        "input": mode,
        "duration": duration,
        "targets": "+".join(targets),
        "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "status": "planned",
        "purpose": purpose,
    })

    # Keep the existing 01-06 experiment artifacts untouched.
    if phase != "07_cross_validation":
        continue

    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(base_config)
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})

    cfg["model"]["type"] = model
    cfg["model"]["targets"] = targets
    cfg["data"]["mode"] = mode
    cfg["data"]["filter_duration"] = duration

    # Autoencoder-specific configuration
    if trainer == "autoencoder":
        cfg.setdefault("autoencoder", {})
        cfg["autoencoder"]["enabled"] = True
        cfg["autoencoder"]["output_dir"] = "./Baseline/results"

    # Cross-validation-specific configuration
    if trainer == "cross_validate":
        cfg["training"]["n_folds"] = N_FOLDS
        cfg["training"].setdefault("seed", CV_SEED)

        # Support either a boolean or a dictionary for early_stopping.
        early_stopping = cfg["training"].get("early_stopping", {})
        if isinstance(early_stopping, bool):
            early_stopping = {"enabled": early_stopping}
        elif not isinstance(early_stopping, dict):
            early_stopping = {}

        early_stopping.setdefault("enabled", True)
        early_stopping.setdefault("patience", 15)
        early_stopping.setdefault("min_delta", 1e-4)
        early_stopping.setdefault("restore_best_weights", True)
        cfg["training"]["early_stopping"] = early_stopping

    # Keep target-specific settings consistent with the chosen targets.
    for key in ("loss_functions", "loss_weights"):
        if key in cfg["model"]:
            missing = [
                target
                for target in targets
                if target not in cfg["model"][key]
            ]
            if missing:
                print(f"WARNING {name}: add {key} for targets {missing}")

            cfg["model"][key] = {
                target: cfg["model"][key][target]
                for target in targets
                if target in cfg["model"][key]
            }

    if config_path.exists():
        print(f"Preserving existing config: {config_path}")
    else:
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    if readme_path.exists():
        print(f"Preserving existing README: {readme_path}")
    else:
        readme_path.write_text(
            f"# {name}\n\n"
            f"- **Phase:** {phase}\n"
            f"- **Trainer:** {trainer}\n"
            f"- **Model:** {model}\n"
            f"- **Input:** {mode}\n"
            f"- **Duration:** {duration}\n"
            f"- **Targets:** {', '.join(targets) if targets else 'None'}\n"
            f"- **Purpose:** {purpose}\n\n"
            "## Before running\n"
            "- Verify loss functions and target weights in config.yaml.\n"
            "- Verify the manifest and input paths.\n"
            "- Verify that the intended data split is KIC-level.\n"
            "- Record any deviations from the planned experiment.\n\n"
            "## Results\n"
            "- Run directory:\n"
            "- Best epoch / fold results:\n"
            "- Evaluation metrics:\n"
            "- Notes:\n"
        )

plan_path = EXPERIMENTS_DIR / "experiment_plan.csv"
plan_path.parent.mkdir(parents=True, exist_ok=True)

with open(plan_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=plan_rows[0].keys())
    writer.writeheader()
    writer.writerows(plan_rows)

# =========================================================
# Interpretability plan
# This remains separate from the training experiment plan.
# It is handled by run_interpretation_array.py and writes
# its outputs under Baseline/interpretation/.
# =========================================================

INTERPRETABILITY_DIR = ROOT / "interpretation"
INTERPRETABILITY_DIR.mkdir(parents=True, exist_ok=True)

interpretation_experiments = [
    {
        "analysis_id": "I0_cnn_lstm_lc_regression_ig",
        "model": "cnn_lstm",
        "mode": "lc",
        "duration": "80d",
        "targets": "nu_max+delta_nu",
        "run_dir": (
            "Baseline/results/"
            "cnn_lstm_lc_80d_nu_max_delta_nu_20261020_120605_552398"
        ),
        "method": "ig",
        "max_samples": 10,
        "status": "planned",
    },

        {
        "analysis_id": "I0_cnn_lstm_lc_regression_sg",
        "model": "cnn_lstm",
        "mode": "lc",
        "duration": "80d",
        "targets": "nu_max+delta_nu",
        "run_dir": (
            "Baseline/results/"
            "cnn_lstm_lc_80d_nu_max_delta_nu_20261020_120605_552398"
        ),
        "method": "smoothgrad",
        "max_samples": 10,
        "status": "planned",
    },
    {
        "analysis_id": "I1_astroconformer_v2_psd_regression_ig",
        "model": "astroconformer_v2",
        "mode": "psd",
        "duration": "80d",
        "targets": "nu_max+delta_nu",
        "run_dir": (
            "Baseline/results/"
            "astroconformer_v2_psd_80d_nu_max_delta_nu_20261020_123350_350217"
        ),
        "method": "ig",
        "max_samples": 10,
        "status": "planned",
    },
        {
        "analysis_id": "I1_astroconformer_v2_psd_regression_sg",
        "model": "astroconformer_v2",
        "mode": "psd",
        "duration": "80d",
        "targets": "nu_max+delta_nu",
        "run_dir": (
            "Baseline/results/"
            "astroconformer_v2_psd_80d_nu_max_delta_nu_20261020_123350_350217"
        ),
        "method": "smoothgrad",
        "max_samples": 10,
        "status": "planned",
    },
    {
        "analysis_id": "I2_resnet1d_lc_multitask_ig",
        "model": "resnet1d",
        "mode": "lc",
        "duration": "80d",
        "targets": "nu_max+delta_nu+class",
        "run_dir": (
            "Baseline/results/"
            "resnet1d_lc_80d_nu_max_delta_nu_class_"
            "20261018_185345_104527"
        ),
        "method": "ig",
        "max_samples": 10,
        "status": "planned",
    },
        {
        "analysis_id": "I2_resnet1d_lc_multitask_ig",
        "model": "resnet1d",
        "mode": "lc",
        "duration": "80d",
        "targets": "nu_max+delta_nu+class",
        "run_dir": (
            "Baseline/results/"
            "resnet1d_lc_80d_nu_max_delta_nu_class_"
            "20261018_185345_104527"
        ),
        "method": "smoothgrad",
        "max_samples": 10,
        "status": "planned",
    },
]

interpretation_plan_path = (
    INTERPRETABILITY_DIR / "interpretation_plan.csv"
)

with open(interpretation_plan_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=interpretation_experiments[0].keys(),
    )
    writer.writeheader()
    writer.writerows(interpretation_experiments)

# =========================================================
# Summary
# =========================================================

print("Preserved existing experiment configs and READMEs for phases 01-06.")
print(f"Created or preserved {len(cv_experiments)} cross-validation experiment folders.")
print(f"Experiment plan: {plan_path}")
print(f"Interpretability plan: {interpretation_plan_path}")
print("Review the generated configs before submitting jobs.")

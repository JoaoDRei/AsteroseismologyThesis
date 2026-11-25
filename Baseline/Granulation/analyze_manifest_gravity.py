#!/usr/bin/env python3
"""Compare seismic and spectroscopic surface gravities in the manifest."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "Baseline" / "manifest.csv"
OUTPUT_DIR = ROOT / "results" / "gravity_comparison_v4"
SOLAR_NUMAX = 3090.0
SOLAR_TEFF = 5772.0


def load_metadata():
    raw = pd.read_csv(MANIFEST, low_memory=False)
    seismic_column = (
        "logg_seis" if "logg_seis" in raw.columns else "logg"
    )
    required = {
        "kic", "nu_max", "teff", "metallicity", "mass",
        "logg_spec", seismic_column,
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    data = pd.DataFrame({
        "kic": raw["kic"],
        "nu_max": raw["nu_max"],
        "teff": raw["teff"],
        "metallicity": raw["metallicity"],
        "mass": raw["mass"],
        "logg_seis": raw[seismic_column],
        "logg_spec": raw["logg_spec"],
    })
    data["kic"] = pd.to_numeric(data["kic"], errors="raise").astype("int64")
    for column in ("nu_max", "teff", "mass", "logg_seis", "logg_spec"):
        values = pd.to_numeric(data[column], errors="coerce")
        data[column] = values.where(np.isfinite(values) & (values > 0))
    data["metallicity"] = pd.to_numeric(
        data["metallicity"], errors="coerce"
    ).where(lambda values: np.isfinite(values) & (values > -999))
    for column in ("logg_seis", "logg_spec"):
        data[column] = data[column].where(data[column].between(0, 6))

    data = data.groupby("kic", as_index=False).median(numeric_only=True)
    data = data.dropna(subset=[
        "nu_max", "teff", "metallicity", "mass", "logg_seis", "logg_spec",
    ]).copy()
    data["delta_logg"] = data["logg_spec"] - data["logg_seis"]
    data["N"] = np.log10(data["nu_max"] / SOLAR_NUMAX)
    data["T"] = np.log10(data["teff"] / SOLAR_TEFF)
    data["M"] = np.log10(data["mass"])
    return data


def main():
    data = load_metadata()
    if len(data) < 10:
        raise ValueError(
            f"Only {len(data)} complete stars; too few for the diagnostic."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "gravity_comparison_by_kic.csv", index=False)

    spearman = data[
        ["delta_logg", "metallicity", "teff", "mass", "nu_max"]
    ].corr(method="spearman")["delta_logg"]
    design = sm.add_constant(data[["N", "T", "M", "metallicity"]])
    regression = sm.OLS(data["delta_logg"], design).fit(cov_type="HC3")

    lines = [
        f"Complete KICs: {len(data)}",
        "",
        "Spectroscopic minus seismic log g (dex):",
        data["delta_logg"].describe().to_string(),
        "",
        "Spearman correlations with gravity difference:",
        spearman.to_string(),
        "",
        "HC3 OLS: delta_logg ~ log10(nu_max/3090) + "
        "log10(teff/5772) + log10(mass) + metallicity",
        regression.summary().as_text(),
    ]
    report = "\n".join(lines)
    (OUTPUT_DIR / "gravity_comparison.txt").write_text(report)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].hist(data["delta_logg"], bins=80)
    axes[0, 0].set_xlabel("Spectroscopic - seismic log g (dex)")
    axes[0, 0].set_ylabel("Number of stars")
    for ax, column in zip(
        axes.flat[1:],
        ("metallicity", "teff", "mass", "nu_max", "logg_seis"),
    ):
        ax.scatter(data[column], data["delta_logg"], s=2, alpha=0.15)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xlabel(column)
        ax.set_ylabel("Spectroscopic - seismic log g (dex)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gravity_comparison.png", dpi=200)
    plt.close(fig)

    print(f"Saved: {OUTPUT_DIR}")
    print(f"Complete KICs: {len(data)}")
    print("\nSpectroscopic minus seismic log g (dex):")
    print(data["delta_logg"].describe().to_string())
    print("\nSpearman correlations:")
    print(spearman.to_string())
    print("\nHC3 OLS summary:")
    print(regression.summary())


if __name__ == "__main__":
    main()

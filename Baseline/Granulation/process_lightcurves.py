#!/usr/bin/env python3
"""
Create physically-scaled PSDs from corrected KEPSEISMIC FITS light curves.

Important:
- Does NOT divide flux by its standard deviation.
- Converts frequency explicitly to microhertz.
- Keeps Lightkurve/Astropy unit information in a JSON sidecar.
- Does not sigma-clip by default: clipping can remove real stellar variability.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import lightkurve as lk
import astropy.units as u
from tqdm import tqdm

warnings.filterwarnings("ignore", category=lk.utils.LightkurveWarning)


def extract_kic(path: Path) -> int | None:
    match = re.search(r"kplr(\d{9})", path.name)
    return int(match.group(1)) if match else None


def process_one(path: Path, out_dir: Path, min_freq: float, max_freq: float,
                sigma_clip: bool = False) -> dict | None:
    kic = extract_kic(path)
    if kic is None:
        return None

    lc = lk.read(path).remove_nans()
    if len(lc) < 100:
        raise ValueError(f"Too few valid cadences: {len(lc)}")

    # Preserve the flux unit and scale. Subtracting a constant only removes DC.
    flux = lc.flux
    time = lc.time
    if sigma_clip:
        # Optional and deliberately off by default.
        from astropy.stats import sigma_clip as astropy_sigma_clip
        clipped = astropy_sigma_clip(flux.value, sigma=8, cenfunc="median")
        keep = ~np.asarray(clipped.mask)
        time = time[keep]
        flux = flux[keep]
    flux = flux - np.nanmedian(flux.value) * flux.unit

    clean_lc = lk.LightCurve(time=time, flux=flux)
    pg = clean_lc.to_periodogram(method="lombscargle", normalization="psd")

    # Lightkurve frequencies are commonly expressed in 1/day for Kepler data.
    # Convert from the Quantity rather than assuming the numeric unit.
    freq_uHz = pg.frequency.to(u.uHz).value
    power = pg.power.value

    valid = (
        np.isfinite(freq_uHz) & np.isfinite(power) &
        (freq_uHz >= min_freq) & (freq_uHz <= max_freq) & (power > 0)
    )
    freq_uHz = np.asarray(freq_uHz[valid], dtype=np.float64)
    power = np.asarray(power[valid], dtype=np.float64)
    order = np.argsort(freq_uHz)
    freq_uHz, power = freq_uHz[order], power[order]
    if len(freq_uHz) < 100:
        raise ValueError("Fewer than 100 PSD bins in requested frequency interval")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"kic{kic:09d}_{path.stem}"
    npz_path = out_dir / f"{stem}.npz"
    meta_path = out_dir / f"{stem}.json"

    np.savez_compressed(npz_path, frequency_uHz=freq_uHz, power=power)
    metadata = {
        "kic": kic,
        "source_fits": str(path),
        "n_cadences": int(len(time)),
        "frequency_unit": "uHz",
        "power_unit": str(pg.power.unit),
        "flux_unit": str(lc.flux.unit),
        "periodogram_method": "lightkurve LombScargle",
        "periodogram_normalization": "psd",
        "flux_standardized": False,
        "median_subtracted": True,
        "sigma_clip_enabled": bool(sigma_clip),
        "frequency_min_uHz": float(freq_uHz.min()),
        "frequency_max_uHz": float(freq_uHz.max()),
        "n_frequency_bins": int(len(freq_uHz)),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="./data/psd_physical")
    p.add_argument("--pattern", default="*cor-filt-inp.fits")
    p.add_argument("--min-frequency", type=float, default=0.1, help="microhertz")
    p.add_argument("--max-frequency", type=float, default=283.0, help="microhertz")
    p.add_argument("--sigma-clip", action="store_true",
                   help="Optional 8-sigma clipping; off by default")
    args = p.parse_args()

    root = (
        Path(__file__).resolve().parent
        / "kepseismic_data_APOKASC3"
        / "mastDownload"
        / "HLSP"
    )
    out = Path(args.output)
    fits_files = sorted(root.rglob(args.pattern))
    if not fits_files:
        raise SystemExit(f"No files matching {args.pattern!r} under {root}")

    success, failures = [], []
    for fp in tqdm(fits_files, desc="Building physical PSDs"):
        try:
            result = process_one(fp, out, args.min_frequency, args.max_frequency,
                                 args.sigma_clip)
            if result:
                success.append(result)
        except Exception as exc:
            failures.append({"file": str(fp), "error": repr(exc)})

    (out / "processing_summary.json").write_text(json.dumps({
        "processed": len(success), "failed": len(failures),
        "frequency_min_uHz": args.min_frequency,
        "frequency_max_uHz": args.max_frequency,
        "sigma_clip": args.sigma_clip,
        "failures": failures,
    }, indent=2))
    print(f"Processed: {len(success)}; failed: {len(failures)}")
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()

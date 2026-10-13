#!/usr/bin/env python3

"""
Benchmark-star search for Kepler asteroseismic ML validation.

Designed for ~9,000 unique KICs.

Main strategy
-------------
1. Load unique KICs from Baseline/manifest.csv.
2. Download each VizieR catalogue ONCE.
3. Match all requested KICs locally.
4. Rank stars locally using catalogue evidence and pipeline consistency.
5. Select only the highest-priority stars for ADS.
6. Resolve SIMBAD only for those ADS candidates.
7. Perform ONE combined ADS query per candidate using:
       KIC + SIMBAD main_id + selected aliases
8. Classify the returned papers conservatively.
9. Write:
       Baseline/results/benchmark/benchmark_stars.csv
       Baseline/results/benchmark/benchmark_references.csv
       Baseline/results/benchmark/benchmark_catalogue_hits.csv

Usage
-----
Run the default search:

    python benchmark_search.py

Search ADS for only 1000 candidates:

    python benchmark_search.py --ads-candidates 1000

Search ADS for 2000:

    python benchmark_search.py --ads-candidates 2000

Skip ADS entirely:

    python benchmark_search.py --ads-candidates 0

Search ADS for every star:

    python benchmark_search.py --ads-candidates -1

Force fresh ADS/SIMBAD/VizieR queries:

    python benchmark_search.py --refresh-cache

Use more ADS workers:

    python benchmark_search.py --ads-candidates 1000 --ads-workers 4

Limit the number of stars for a test run:

    python benchmark_search.py --max-stars 100

Environment
-----------
Requires:

    ADS_API_TOKEN

Example:

    export ADS_API_TOKEN="your-token"

Dependencies
------------
numpy
pandas
requests
astroquery
astropy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

# =============================================================================
# BENCHMARK-CATALOGUE OUTPUT METRICS
# =============================================================================
#
# The script ranks Kepler stars as potential "benchmark stars" for validating
# our ML predictions of nu_max (and, where available, Delta_nu).
#
# IMPORTANT:
# A high score does NOT automatically mean that a star has an independent
# measurement of nu_max. Catalogue membership and literature evidence are
# deliberately kept conceptually separate.
#
# ---------------------------------------------------------------------------
# VIZIER / CATALOGUE METRICS
# ---------------------------------------------------------------------------
#
# vizier_catalogue_count
#     Number of distinct VizieR catalogues in which this KIC appears.
#     More catalogues generally means the star has been studied/characterized
#     more extensively.
#
# vizier_score
#     Weighted score from the VizieR catalogues. Catalogues containing
#     particularly useful asteroseismic information receive larger weights.
#
#     Rough interpretation:
#       0       = no relevant catalogue
#       1-4     = weak/general catalogue evidence
#       5-10    = useful asteroseismic catalogue evidence
#       >10     = appears in several strong/relevant catalogues
#
#     A high vizier_score is useful for selecting stars, but is NOT by itself
#     proof of an independent literature measurement: several catalogues may
#     ultimately derive their values from related pipelines/data.
#
# catalogue_level
#     Separate measure of catalogue strength:
#
#       0 = no relevant catalogue
#       1 = catalogue evidence exists
#       2 = at least one strong asteroseismic catalogue
#       3 = at least two strong asteroseismic catalogues
#
# ---------------------------------------------------------------------------
# PIPELINE CONSISTENCY
# ---------------------------------------------------------------------------
#
# numax_lit_sources
#     Number of distinct literature/catalogue sources providing a nu_max
#     value used by the script.
#
#     Larger is generally better because it allows us to check whether
#     independent pipelines agree on nu_max.
#
# numax_lit_std
#     Standard deviation of the available nu_max values.
#
#     LOW values are GOOD: different measurements agree.
#
#     As a rough guide:
#       < 0.5       = very good agreement
#       0.5-1       = good/moderate agreement
#       1-2         = noticeable disagreement
#       > 2         = substantial disagreement; investigate manually
#
#     WARNING: this is an absolute standard deviation in microHz, so its
#     usefulness depends somewhat on the star's nu_max. A relative scatter
#     would be preferable for comparing stars across the full nu_max range.
#
# ---------------------------------------------------------------------------
# ADS / LITERATURE METRICS
# ---------------------------------------------------------------------------
#
# ads_n_papers
#     Number of ADS papers returned for the star after the combined KIC,
#     SIMBAD name and alias search.
#
#     This is mainly a measure of how much literature was found. MORE is not
#     necessarily better: many papers may be irrelevant to asteroseismology.
#
# ads_relevant_papers
#     Number of papers classified as containing useful star-specific
#     astrophysical evidence (asteroseismology, dynamics, interferometry,
#     etc.).
#
# ads_strong_papers
#     Number of papers containing particularly strong benchmark evidence,
#     such as:
#
#       - individual oscillation modes / peak-bagging
#       - detailed asteroseismic forward modelling
#       - star-specific dynamical mass/radius constraints
#       - interferometric radius/angular-diameter constraints
#
#     This is one of the most important metrics for finding genuinely
#     well-studied benchmark stars.
#
# literature_score
#     Weighted score from the classified ADS papers.
#
#     Higher is better, but the score is capped so that a star with many
#     mediocre papers cannot dominate a star with a small number of excellent
#     papers.
#
#     Rough interpretation:
#       0          = no useful literature found
#       1-4        = weak evidence
#       5-11       = meaningful literature evidence
#       >=12       = strong literature evidence
#       >=20       = exceptionally well-supported candidate
#
# ---------------------------------------------------------------------------
# FINAL BENCHMARK METRICS
# ---------------------------------------------------------------------------
#
# benchmark_score
#     Compatibility score = vizier_score + literature_score.
#
#     Useful for ranking candidates, but should NOT be interpreted as a
#     physical quality measure. A high score can result from catalogue
#     coverage rather than genuinely independent measurements.
#
# benchmark_level
#     Main qualitative classification:
#
#       0 = no useful benchmark evidence
#       1 = some catalogue/literature evidence
#       2 = good candidate; useful catalogue or literature support
#       3 = strong candidate; at least one strong literature paper
#           (or equivalent strong evidence)
#       4 = excellent benchmark candidate; multiple strong papers and/or
#           very substantial independent literature support
#
# ---------------------------------------------------------------------------
# WHAT MAKES A STAR A GOOD BENCHMARK?
# ---------------------------------------------------------------------------
#
# Ideally we want a star satisfying several of the following:
#
#   1. Multiple independent nu_max / Delta_nu measurements
#   2. Small disagreement between those measurements
#   3. Individual-mode or peak-bagging analysis
#   4. Detailed asteroseismic modelling
#   5. Independent dynamical mass/radius constraints
#   6. Interferometric radius/diameter measurements
#   7. Good-quality, long-duration Kepler observations
#   8. Well-established stellar parameters from independent methods
#
# The BEST benchmark stars are therefore not simply the stars with the
# highest benchmark_score. They are stars for which we can trace an actual,
# preferably independent, chain of evidence supporting the published
# nu_max / Delta_nu values.
#
# The final benchmark list should therefore be manually inspected using
# benchmark_references.csv before being used as a scientific validation set.
# =============================================================================
# =============================================================================
# Paths
# =============================================================================

BASELINE_DIR = Path("Baseline")

MANIFEST_PATH = BASELINE_DIR / "manifest.csv"

STARS_OUTPUT = BASELINE_DIR / "results" / "benchmark" / "benchmark_stars.csv"
REFERENCES_OUTPUT = BASELINE_DIR / "results" / "benchmark" / "benchmark_references.csv"
CATALOGUE_OUTPUT = BASELINE_DIR / "results" / "benchmark" / "benchmark_catalogue_hits.csv"

CACHE_DIR = BASELINE_DIR / "results" / "benchmark" / "benchmark_cache"
ADS_CACHE_DIR = CACHE_DIR / "ads"
SIMBAD_CACHE_DIR = CACHE_DIR / "simbad"
VIZIER_CACHE_DIR = CACHE_DIR / "vizier"


# =============================================================================
# VizieR catalogue definitions
# =============================================================================

VIZIER_CATALOGUES = {
    "J/ApJ/835/173": {
        "name": "Kepler LEGACY",
        "weight": 8,
        "kind": "strong_asteroseismic",
        "kic_columns": ["KIC", "KICID", "KIC_ID"],
    },

    "J/ApJ/927/167": {
        "name": "Li+2022 Kepler RGB modelling",
        "weight": 7,
        "kind": "strong_asteroseismic",
        "kic_columns": ["KIC", "KICID", "KIC_ID"],
    },

    "J/ApJS/239/32": {
        "name": "APOKASC-2",
        "weight": 2,
        "kind": "asteroseismic_catalogue",
        "kic_columns": ["KIC", "KICID", "KIC_ID"],
    },

    "J/ApJS/236/42": {
        "name": "Yu+2018 Kepler red giants",
        "weight": 2,
        "kind": "asteroseismic_catalogue",
        "kic_columns": ["KIC", "KICID", "KIC_ID"],
    },

    "J/ApJ/838/115": {
        "name": "Arentoft+2017 individual red giants",
        "weight": 10,
        "kind": "individual_modes",
        "kic_columns": ["KIC", "KICID", "KIC_ID"],
    },

    "J/ApJS/229/30": {
        "name": "Kepler DR25 stellar properties",
        "weight": 1,
        "kind": "general_catalogue",
        "kic_columns": ["KIC", "KICID", "KIC_ID"],
    },
}


STRONG_VIZIER_CATALOGUES = {
    "J/ApJ/835/173",
    "J/ApJ/927/167",
    "J/ApJ/838/115",
}


ASTERoseismic_VIZIER_CATALOGUES = {
    "J/ApJ/835/173",
    "J/ApJ/927/167",
    "J/ApJS/239/32",
    "J/ApJS/236/42",
    "J/ApJ/838/115",
}


# =============================================================================
# ADS evidence definitions
# =============================================================================

# The classifier deliberately distinguishes:
#
#   strong evidence
#       individual modes
#       detailed asteroseismic modelling
#       dynamical mass/radius
#       interferometric radius/angular diameter
#
#   weaker evidence
#       explicit global νmax/Δν analysis
#       independent νmax
#       Δν
#       spectroscopy/Gaia/activity
#
# Generic mentions of "eclipsing binary" are NOT enough for dynamical evidence.

ADS_EVIDENCE = {
    "individual_modes": {
        "weight": 10,
        "strong": True,
        "patterns": [
            r"\bindividual\s+(?:oscillation|p[- ]?mode|mode)\s+frequenc",
            r"\bmode\s+frequenc(?:y|ies)\s+(?:were|was)\s+(?:measured|determined)",
            r"\bpeak[- ]bagging\b",
            r"\bpeakbagging\b",
            r"\bmode\s+identification\b",
            r"\bfrequency\s+separations?\s+of\s+individual\s+modes\b",
        ],
    },

    "detailed_asteroseismic_modeling": {
        "weight": 8,
        "strong": True,
        "patterns": [
            r"\bdetailed\s+asteroseismic\s+(?:modeling|modelling)\b",
            r"\bforward\s+asteroseismic\s+(?:modeling|modelling)\b",
            r"\bgrid[- ]based\s+asteroseismic\s+(?:modeling|modelling)\b",
            r"\bindividual[- ]star\s+asteroseismic\s+(?:modeling|modelling)\b",
            r"\basteroseismic\s+(?:modeling|modelling)\s+of\s+(?:the\s+)?(?:individual|individuals)\b",
            r"\basteroseismic\s+model\s+(?:was|were)\s+(?:constructed|computed|calculated)\b",
            r"\bforward\s+model(?:ing|ling).{0,80}\basteroseismic\b",
        ],
    },

    "dynamical": {
        "weight": 10,
        "strong": True,
        "patterns": [
            r"\bdynamical\s+mass(?:es)?\b",
            r"\bdynamical\s+radius(?:es)?\b",
            r"\bmass(?:es)?\s+(?:determined|measured|derived)\s+from\s+(?:the\s+)?(?:orbital|binary)\b",
            r"\bradius(?:es)?\s+(?:determined|measured|derived)\s+from\s+(?:the\s+)?(?:orbital|binary)\b",
            r"\borbital\s+(?:solution|analysis).{0,100}\b(?:mass|radius)\b",
            r"\b(?:mass|radius).{0,100}\beclipsing\s+binary\b",
            r"\bbinary\s+(?:mass|radius)\s+(?:determination|measurement)\b",
        ],
    },

    "interferometry": {
        "weight": 10,
        "strong": True,
        "patterns": [
            r"\binterferometric\s+angular\s+diameter\b",
            r"\binterferometric\s+radius\b",
            r"\bangular\s+diameter\s+(?:measured|determined)\s+with\s+(?:interferometr|interferometric)\b",
            r"\bstellar\s+radius\s+(?:measured|determined)\s+interferometrically\b",
            r"\binterferometric\s+observations?.{0,100}\b(?:radius|diameter)\b",
        ],
    },

    "global_asteroseismic_parameters": {
        "weight": 5,
        "strong": False,
        "patterns": [
            r"\basteroseismic\s+scaling\s+relations\b",
            r"\bglobal\s+asteroseismic\s+parameters\b",
            r"\bglobal\s+oscillation\s+parameters\b",
            r"\bfrequency\s+of\s+maximum\s+power.{0,120}\blarge\s+frequency\s+separation\b",
            r"\blarge\s+frequency\s+separation.{0,120}\bfrequency\s+of\s+maximum\s+power\b",
            r"\bnu[_\s-]?max.{0,120}\bdelta[_\s-]?nu\b",
            r"\bdelta[_\s-]?nu.{0,120}\bnu[_\s-]?max\b",
            r"\bnumax.{0,120}\b(?:dnu|delta[_\s-]?nu)\b",
            r"\b(?:dnu|delta[_\s-]?nu).{0,120}\bnumax\b",
        ],
    },

    "independent_numax": {
        "weight": 8,
        "strong": False,
        "patterns": [
            r"\b(?:nu[_\s-]?max|numax|frequency\s+of\s+maximum\s+power).{0,80}\b(?:measured|determined|estimated|derived)\b",
            r"\b(?:measured|determined|estimated|derived).{0,80}\b(?:nu[_\s-]?max|numax|frequency\s+of\s+maximum\s+power)\b",
        ],
    },

    "delta_nu": {
        "weight": 5,
        "strong": False,
        "patterns": [
            r"\b(?:delta[_\s-]?nu|dnu|large\s+frequency\s+separation).{0,80}\b(?:measured|determined|estimated|derived)\b",
            r"\b(?:measured|determined|estimated|derived).{0,80}\b(?:delta[_\s-]?nu|dnu|large\s+frequency\s+separation)\b",
        ],
    },

    "spectroscopy": {
        "weight": 2,
        "strong": False,
        "patterns": [
            r"\bhigh[- ]resolution\s+spectroscop",
            r"\bspectroscopic\s+(?:temperature|metallicity|abundance|parameters?)\b",
            r"\bmetallicity\s+(?:was|were)\s+(?:measured|determined)\b",
        ],
    },

    "gaia": {
        "weight": 2,
        "strong": False,
        "patterns": [
            r"\bgaia\s+(?:parallax|distance)\b",
            r"\bgaia\s+astrometr",
        ],
    },

    "magnetic_activity": {
        "weight": 1,
        "strong": False,
        "patterns": [
            r"\bmagnetic\s+activity\b",
            r"\bstellar\s+activity\b",
            r"\bactivity\s+cycle\b",
            r"\brotation\s+period\b",
        ],
    },
}


# =============================================================================
# Utility functions
# =============================================================================

def ensure_directories() -> None:
    """Create output/cache directories."""
    for path in [
        BASELINE_DIR,
        CACHE_DIR,
        ADS_CACHE_DIR,
        SIMBAD_CACHE_DIR,
        VIZIER_CACHE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_kic(value: Any) -> Optional[str]:
    """
    Normalize a KIC value to a plain integer-like string.

    Examples:
        4991033       -> "4991033"
        "4991033"     -> "4991033"
        "KIC 4991033" -> "4991033"
        4991033.0     -> "4991033"
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()

    text = re.sub(r"^KIC\s*", "", text, flags=re.IGNORECASE)

    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]

    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except Exception:
        pass

    digits = re.search(r"\d+", text)
    if digits:
        return digits.group(0)

    return None


def safe_json(value: Any) -> Any:
    """Convert NumPy/Astropy/Pandas values into JSON-safe values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]

    return value


def save_json(path: Path, obj: Any) -> None:
    """Write JSON atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            safe_json(obj),
            f,
            ensure_ascii=False,
            indent=2,
        )

    tmp.replace(path)


def load_json(path: Path) -> Optional[Any]:
    """Load JSON if it exists and is valid."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def first_existing_column(
    columns: Iterable[str],
    candidates: Sequence[str],
) -> Optional[str]:
    """
    Find a column using case-insensitive matching.
    """
    lookup = {str(c).lower(): str(c) for c in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []

    for value in values:
        value = str(value).strip()

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


# =============================================================================
# Manifest / pipeline information
# =============================================================================

def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Could not find manifest: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    if "kic" not in df.columns:
        raise ValueError(
            "manifest.csv does not contain a 'kic' column."
        )

    df["kic_normalized"] = df["kic"].map(normalize_kic)

    df = df[df["kic_normalized"].notna()].copy()

    # One row per unique KIC.
    df = df.drop_duplicates("kic_normalized", keep="first")

    df["kic"] = df["kic_normalized"].astype(str)

    df = df.drop(columns=["kic_normalized"])

    return df.reset_index(drop=True)


def add_pipeline_scatter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate agreement among the available pipeline νmax and Δν values.
    """

    numax_columns = [
        "nu_max",
        "nu_max_hon",
        "nu_max_syd",
        "nu_max_a2z",
        "nu_max_dia",
    ]

    dnu_columns = [
        "delta_nu",
        "delta_nu_syd",
        "delta_nu_a2z",
        "delta_nu_dia",
    ]

    numax_columns = [
        c for c in numax_columns if c in df.columns
    ]

    dnu_columns = [
        c for c in dnu_columns if c in df.columns
    ]

    def statistics(row: pd.Series, columns: List[str]) -> pd.Series:
        values = pd.to_numeric(
            row[columns],
            errors="coerce",
        ).dropna()

        values = values[values > 0]

        if len(values) == 0:
            return pd.Series(
                {
                    "n_sources": 0,
                    "median": np.nan,
                    "std": np.nan,
                    "relative_std": np.nan,
                }
            )

        median = float(values.median())
        std = float(values.std(ddof=0)) if len(values) > 1 else 0.0

        relative_std = (
            std / abs(median)
            if median != 0
            else np.nan
        )

        return pd.Series(
            {
                "n_sources": int(len(values)),
                "median": median,
                "std": std,
                "relative_std": relative_std,
            }
        )

    if numax_columns:
        numax_stats = df.apply(
            lambda row: statistics(row, numax_columns),
            axis=1,
        )

        df["numax_pipeline_sources"] = (
            numax_stats["n_sources"]
        )

        df["numax_pipeline_median"] = (
            numax_stats["median"]
        )

        df["numax_pipeline_std"] = (
            numax_stats["std"]
        )

        df["numax_pipeline_relative_std"] = (
            numax_stats["relative_std"]
        )

    else:
        df["numax_pipeline_sources"] = 0
        df["numax_pipeline_median"] = np.nan
        df["numax_pipeline_std"] = np.nan
        df["numax_pipeline_relative_std"] = np.nan

    if dnu_columns:
        dnu_stats = df.apply(
            lambda row: statistics(row, dnu_columns),
            axis=1,
        )

        df["dnu_pipeline_sources"] = (
            dnu_stats["n_sources"]
        )

        df["dnu_pipeline_median"] = (
            dnu_stats["median"]
        )

        df["dnu_pipeline_std"] = (
            dnu_stats["std"]
        )

        df["dnu_pipeline_relative_std"] = (
            dnu_stats["relative_std"]
        )

    else:
        df["dnu_pipeline_sources"] = 0
        df["dnu_pipeline_median"] = np.nan
        df["dnu_pipeline_std"] = np.nan
        df["dnu_pipeline_relative_std"] = np.nan

    return df


# =============================================================================
# VizieR
# =============================================================================

def vizier_cache_path(catalogue_id: str) -> Path:
    filename = (
        sanitize_filename(catalogue_id)
        + "_filtered.json"
    )

    return VIZIER_CACHE_DIR / filename


def extract_filtered_vizier_rows(
    table: Any,
    catalogue_id: str,
    catalogue_config: Dict[str, Any],
    kic_set: set,
) -> List[Dict[str, Any]]:
    """
    Extract only rows belonging to the requested KICs.

    This is important: we NEVER accept a VizieR row merely because
    the query returned it. The KIC in the returned row must match
    one of the requested KICs exactly after normalization.
    """

    try:
        table_df = table.to_pandas()
    except Exception:
        return []

    kic_column = first_existing_column(
        table_df.columns,
        catalogue_config["kic_columns"],
    )

    if kic_column is None:
        return []

    normalized = table_df[kic_column].map(normalize_kic)

    mask = normalized.isin(kic_set)

    filtered = table_df.loc[mask].copy()
    filtered["_normalized_kic"] = normalized.loc[mask]

    results = []

    for _, row in filtered.iterrows():
        kic = row["_normalized_kic"]

        if kic is None:
            continue

        row_dict = {}

        for column in table_df.columns:
            if column == "_normalized_kic":
                continue

            row_dict[str(column)] = safe_json(row[column])

        results.append(
            {
                "kic": str(kic),
                "catalogue_id": catalogue_id,
                "catalogue_name": catalogue_config["name"],
                "catalogue_kind": catalogue_config["kind"],
                "catalogue_weight": catalogue_config["weight"],
                "catalogue_row": row_dict,
            }
        )

    return results


def bulk_vizier_search(
    kics: Sequence[str],
    refresh_cache: bool = False,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """
    Download each VizieR catalogue ONCE and match all KICs locally.

    Returns
    -------
    hits_by_kic
        KIC -> list of validated VizieR hits.

    status
        catalogue ID -> status message.
    """

    kic_set = set(kics)

    hits_by_kic = {
        kic: []
        for kic in kics
    }

    status = {}

    vizier = Vizier(
        row_limit=-1
    )

    print()
    print("=" * 80)
    print("VIZIER BULK SEARCH")
    print("=" * 80)

    for catalogue_id, config in VIZIER_CATALOGUES.items():

        cache_path = vizier_cache_path(catalogue_id)

        cached = None

        if not refresh_cache:
            cached = load_json(cache_path)

        if cached is not None:
            rows = cached

            print(
                f"[VizieR] {catalogue_id}: "
                f"loaded {len(rows)} cached matching rows"
            )

            status[catalogue_id] = "cache"

        else:
            print(
                f"[VizieR] Downloading {catalogue_id} "
                f"({config['name']}) ..."
            )

            try:
                tables = vizier.get_catalogs(catalogue_id)

            except Exception as exc:
                print(
                    f"[VizieR] ERROR for {catalogue_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

                status[catalogue_id] = (
                    f"error: {type(exc).__name__}: {exc}"
                )

                continue

            rows = []

            for table in tables:
                rows.extend(
                    extract_filtered_vizier_rows(
                        table,
                        catalogue_id,
                        config,
                        kic_set,
                    )
                )

            # Cache only successful catalogue downloads.
            save_json(cache_path, rows)

            print(
                f"[VizieR] {catalogue_id}: "
                f"{len(rows)} validated matching rows"
            )

            status[catalogue_id] = "downloaded"

        # Add cached/downloaded hits.
        for hit in rows:
            kic = normalize_kic(hit.get("kic"))

            if kic not in hits_by_kic:
                continue

            hits_by_kic[kic].append(hit)

    print()
    print("[VizieR] Bulk search finished.")

    return hits_by_kic, status


def unique_vizier_hits(
    hits: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Keep one representative row per catalogue.

    Multiple rows from the same catalogue should NOT artificially
    increase the catalogue score.
    """

    result = []
    seen = set()

    for hit in hits:
        catalogue_id = hit["catalogue_id"]

        if catalogue_id in seen:
            continue

        seen.add(catalogue_id)
        result.append(hit)

    return result


def calculate_vizier_score(
    hits: Sequence[Dict[str, Any]],
) -> Tuple[int, int, int]:
    """
    Returns:
        score,
        number of strong catalogues,
        number of asteroseismic catalogues
    """

    unique_hits = unique_vizier_hits(hits)

    score = 0
    strong_count = 0
    asteroseismic_count = 0

    for hit in unique_hits:
        score += int(hit["catalogue_weight"])

        catalogue_id = hit["catalogue_id"]

        if catalogue_id in STRONG_VIZIER_CATALOGUES:
            strong_count += 1

        if catalogue_id in ASTERoseismic_VIZIER_CATALOGUES:
            asteroseismic_count += 1

    # Preserve the old upper cap.
    score = min(score, 20)

    return (
        score,
        strong_count,
        asteroseismic_count,
    )


def catalogue_identifiers(
    hits: Sequence[Dict[str, Any]],
) -> List[str]:
    """
    Extract useful identifiers already present in VizieR rows.

    These are later included in the combined ADS query.
    """

    preferred_columns = [
        "2MASS",
        "TIC",
        "TYC",
        "GSC",
        "Gaia",
        "GaiaDR2",
        "GaiaDR3",
        "HD",
        "HIP",
        "UCAC",
        "AP",
    ]

    identifiers = []

    for hit in hits:

        row = hit.get("catalogue_row", {})

        # Case-insensitive column lookup.
        lower_to_original = {
            str(key).lower(): key
            for key in row.keys()
        }

        for column in preferred_columns:

            actual = lower_to_original.get(
                column.lower()
            )

            if actual is None:
                continue

            value = row.get(actual)

            if value is None:
                continue

            text = str(value).strip()

            if not text:
                continue

            if text.lower() in {
                "nan",
                "none",
                "masked",
                "--",
            }:
                continue

            identifiers.append(text)

    return unique_preserve_order(identifiers)


# =============================================================================
# Candidate ranking
# =============================================================================

def candidate_priority(
    kic: str,
    star_row: pd.Series,
    hits: Sequence[Dict[str, Any]],
) -> Tuple:
    """
    Rank candidates for ADS.

    Strong VizieR catalogues are deliberately given high priority,
    because they are the most likely to contain stars with detailed
    literature.

    Pipeline agreement is useful but is NOT treated as independent
    literature evidence.
    """

    vizier_score, strong_count, astro_count = (
        calculate_vizier_score(hits)
    )

    numax_sources = int(
        star_row.get(
            "numax_pipeline_sources",
            0,
        )
        or 0
    )

    dnu_sources = int(
        star_row.get(
            "dnu_pipeline_sources",
            0,
        )
        or 0
    )

    numax_relative_std = star_row.get(
        "numax_pipeline_relative_std",
        np.nan,
    )

    if pd.isna(numax_relative_std):
        numax_relative_std = 999.0

    # The tuple is sorted descending.
    #
    # 1. Strong catalogue membership
    # 2. VizieR score
    # 3. Number of asteroseismic catalogues
    # 4. Number of independent νmax pipelines
    # 5. Number of Δν pipelines
    # 6. Better νmax agreement
    #
    return (
        strong_count,
        vizier_score,
        astro_count,
        numax_sources,
        dnu_sources,
        -float(numax_relative_std),
    )


def select_ads_candidates(
    df: pd.DataFrame,
    hits_by_kic: Dict[str, List[Dict[str, Any]]],
    n_candidates: int,
) -> List[str]:
    """
    Select the highest-priority stars for ADS.

    n_candidates:
        0  -> no ADS
        -1 -> all stars
        >0 -> top N stars
    """

    if n_candidates == 0:
        return []

    all_kics = df["kic"].astype(str).tolist()

    if n_candidates < 0:
        return all_kics

    scored = []

    for _, row in df.iterrows():

        kic = str(row["kic"])

        priority = candidate_priority(
            kic,
            row,
            hits_by_kic.get(kic, []),
        )

        scored.append(
            (
                priority,
                kic,
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    selected = [
        kic
        for _, kic in scored[:n_candidates]
    ]

    return selected


# =============================================================================
# SIMBAD
# =============================================================================

def simbad_cache_path(kic: str) -> Path:
    return SIMBAD_CACHE_DIR / f"{kic}.json"


def parse_simbad_result(
    result: Any,
) -> Dict[str, Any]:
    """
    Convert a SIMBAD result table into a simple dictionary.
    """

    if result is None:
        return {
            "found": False,
            "main_id": None,
            "ids": [],
        }

    try:
        if len(result) == 0:
            return {
                "found": False,
                "main_id": None,
                "ids": [],
            }
    except Exception:
        return {
            "found": False,
            "main_id": None,
            "ids": [],
        }

    row = result[0]

    def get_column(name: str) -> Any:
        try:
            return row[name]
        except Exception:
            return None

    main_id = get_column("main_id")

    if main_id is not None:
        main_id = str(main_id).strip()

    ids_value = get_column("ids")

    ids = []

    if ids_value is not None:
        if isinstance(ids_value, bytes):
            ids_value = ids_value.decode(
                errors="ignore"
            )

        text = str(ids_value)

        # SIMBAD usually separates identifiers by "|".
        for identifier in text.split("|"):
            identifier = identifier.strip()

            if identifier:
                ids.append(identifier)

    return {
        "found": True,
        "main_id": main_id,
        "ids": unique_preserve_order(ids),
    }


def query_simbad_one(
    kic: str,
    refresh_cache: bool = False,
) -> Dict[str, Any]:

    cache_path = simbad_cache_path(kic)

    if not refresh_cache:

        cached = load_json(cache_path)

        if cached is not None:
            return cached

    custom_simbad = Simbad()

    # Keep the response reasonably small.
    try:
        custom_simbad.add_votable_fields(
            "ids"
        )
    except Exception:
        pass

    result = None

    try:
        result = custom_simbad.query_object(
            f"KIC {kic}"
        )

    except Exception as exc:
        data = {
            "found": False,
            "main_id": None,
            "ids": [],
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

        save_json(cache_path, data)

        return data

    data = parse_simbad_result(result)

    save_json(cache_path, data)

    return data


# =============================================================================
# ADS
# =============================================================================

ADS_URL = (
    "https://api.adsabs.harvard.edu/v1/search/query"
)


def get_ads_token() -> Optional[str]:
    token = os.environ.get(
        "ADS_API_TOKEN"
    )

    if token:
        return token.strip()

    return None


def ads_cache_path(kic: str) -> Path:
    return ADS_CACHE_DIR / f"{kic}.json"


def clean_identifier_for_ads(
    identifier: str,
) -> Optional[str]:
    """
    Prepare an identifier for an ADS quoted search term.
    """

    if identifier is None:
        return None

    text = str(identifier).strip()

    if not text:
        return None

    # Remove problematic quotation characters.
    text = text.replace('"', " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    if not text:
        return None

    return text


def make_ads_query(
    kic: str,
    main_id: Optional[str],
    aliases: Sequence[str],
) -> str:
    """
    Construct ONE ADS query containing several identifiers.

    We search both normal text and the full-text field for the
    identifiers. This avoids doing separate ADS requests for every
    alias.
    """

    identifiers = []

    identifiers.append(
        f"KIC {kic}"
    )

    if main_id:
        identifiers.append(
            main_id
        )

    identifiers.extend(
        aliases
    )

    identifiers = [
        clean_identifier_for_ads(x)
        for x in identifiers
    ]

    identifiers = [
        x for x in identifiers
        if x
    ]

    identifiers = unique_preserve_order(
        identifiers
    )

    # Avoid an enormous ADS query.
    identifiers = identifiers[:6]

    clauses = []

    for identifier in identifiers:

        escaped = identifier.replace(
            '"',
            '\\"',
        )

        clauses.append(
            f'"{escaped}"'
        )

        clauses.append(
            f'full:"{escaped}"'
        )

    clauses = unique_preserve_order(
        clauses
    )

    return "(" + " OR ".join(clauses) + ")"


def ads_request(
    query: str,
    token: str,
    rows: int = 30,
    max_retries: int = 4,
) -> Dict[str, Any]:
    """
    Execute an ADS request with retry/backoff.

    Retries:
        429
        500-599
    """

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    params = {
        "q": query,
        "fl": (
            "bibcode,title,author,year,"
            "abstract,identifier"
        ),
        "rows": rows,
        "sort": "date desc",
    }

    last_error = None

    for attempt in range(max_retries):

        try:
            response = requests.get(
                ADS_URL,
                headers=headers,
                params=params,
                timeout=30,
            )

        except requests.RequestException as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            sleep_time = 2 ** attempt

            time.sleep(
                sleep_time
            )

            continue

        if response.status_code == 200:

            try:
                return response.json()

            except Exception as exc:
                return {
                    "error": (
                        "Invalid JSON from ADS: "
                        f"{exc}"
                    )
                }

        if (
            response.status_code == 429
            or 500 <= response.status_code < 600
        ):

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

            sleep_time = 2 ** attempt

            time.sleep(
                sleep_time
            )

            continue

        # Do not retry authentication or malformed-query errors.
        return {
            "error": (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        }

    return {
        "error": last_error
        or "ADS request failed"
    }


def test_ads(token: str) -> bool:
    """
    Verify the ADS token before launching parallel searches.
    """

    print()
    print("=" * 80)
    print("TESTING ADS CONNECTION")
    print("=" * 80)

    response = ads_request(
        query="asteroseismology",
        token=token,
        rows=1,
    )

    if "error" in response:
        print(
            "[ADS] Connection test FAILED:"
        )

        print(
            response["error"]
        )

        return False

    response_data = response.get(
        "response",
        {},
    )

    num_found = response_data.get(
        "numFound",
        0,
    )

    print(
        f"[ADS] Connection OK. "
        f"Test query returned {num_found} result(s)."
    )

    return True


# =============================================================================
# ADS evidence classification
# =============================================================================

def combined_paper_text(
    paper: Dict[str, Any],
) -> str:

    title = paper.get(
        "title",
        "",
    )

    abstract = paper.get(
        "abstract",
        "",
    )

    if isinstance(title, list):
        title = " ".join(
            str(x)
            for x in title
        )

    if isinstance(abstract, list):
        abstract = " ".join(
            str(x)
            for x in abstract
        )

    return (
        str(title)
        + "\n"
        + str(abstract)
    ).lower()


def classify_paper(
    paper: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify a paper conservatively.

    A paper may trigger several evidence categories, but its
    contribution is capped so that a single paper cannot dominate
    the benchmark score simply because it contains many keywords.
    """

    text = combined_paper_text(
        paper
    )

    evidence = []

    for category, config in ADS_EVIDENCE.items():

        matched = False

        for pattern in config["patterns"]:

            try:
                if re.search(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                ):
                    matched = True
                    break

            except re.error:
                continue

        if matched:

            evidence.append(
                {
                    "category": category,
                    "weight": config["weight"],
                    "strong": config["strong"],
                }
            )

    if not evidence:

        return {
            "relevant": False,
            "manual_review": True,
            "strong": False,
            "evidence_categories": [],
            "evidence_score": 0,
        }

    # Highest evidence category determines whether this is strong.
    strong = any(
        item["strong"]
        for item in evidence
    )

    # Add the strongest category plus smaller complementary
    # categories, but cap one paper at 12.
    weights = sorted(
        (
            item["weight"]
            for item in evidence
        ),
        reverse=True,
    )

    score = min(
        sum(weights),
        12,
    )

    categories = [
        item["category"]
        for item in evidence
    ]

    return {
        "relevant": True,
        "manual_review": False,
        "strong": strong,
        "evidence_categories": categories,
        "evidence_score": score,
    }


def classify_ads_papers(
    papers: Sequence[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    int,
    int,
    int,
]:
    """
    Returns:
        references
        relevant_count
        manual_review_count
        strong_count
    """

    references = []

    relevant_count = 0
    manual_review_count = 0
    strong_count = 0

    for paper in papers:

        classification = classify_paper(
            paper
        )

        title = paper.get(
            "title",
            "",
        )

        if isinstance(title, list):
            title = " ".join(
                str(x)
                for x in title
            )

        authors = paper.get(
            "author",
            [],
        )

        if isinstance(authors, str):
            authors = [authors]

        references.append(
            {
                "bibcode": paper.get(
                    "bibcode"
                ),
                "title": title,
                "authors": "; ".join(
                    str(x)
                    for x in authors
                ),
                "year": paper.get(
                    "year"
                ),
                "classification": (
                    "relevant"
                    if classification["relevant"]
                    else "manual_review"
                ),
                "manual_review": int(
                    classification["manual_review"]
                ),
                "strong": int(
                    classification["strong"]
                ),
                "evidence_categories": ";".join(
                    classification[
                        "evidence_categories"
                    ]
                ),
                "evidence_score": (
                    classification[
                        "evidence_score"
                    ]
                ),
            }
        )

        if classification["relevant"]:
            relevant_count += 1

            if classification["strong"]:
                strong_count += 1

        else:
            manual_review_count += 1

    return (
        references,
        relevant_count,
        manual_review_count,
        strong_count,
    )


def search_ads_candidate(
    kic: str,
    main_id: Optional[str],
    aliases: Sequence[str],
    token: str,
    refresh_cache: bool = False,
) -> Dict[str, Any]:
    """
    Search ADS once for one candidate.

    Raw ADS papers are cached so that changes to the classifier
    do not require new ADS queries.
    """

    cache_path = ads_cache_path(kic)

    cached = None

    if not refresh_cache:
        cached = load_json(
            cache_path
        )

    if cached is not None:

        papers = cached.get(
            "papers",
            [],
        )

        query = cached.get(
            "query",
            "",
        )

        from_cache = True

    else:

        query = make_ads_query(
            kic=kic,
            main_id=main_id,
            aliases=aliases,
        )

        response = ads_request(
            query=query,
            token=token,
            rows=30,
        )

        if "error" in response:

            return {
                "kic": kic,
                "query": query,
                "papers": [],
                "references": [],
                "ads_n_papers": 0,
                "ads_relevant_papers": 0,
                "ads_manual_review_papers": 0,
                "ads_strong_papers": 0,
                "literature_score": 0,
                "error": response["error"],
                "from_cache": False,
            }

        response_data = response.get(
            "response",
            {},
        )

        papers = response_data.get(
            "docs",
            [],
        )

        cache_data = {
            "query": query,
            "papers": papers,
        }

        save_json(
            cache_path,
            cache_data,
        )

        from_cache = False

    (
        references,
        relevant_count,
        manual_count,
        strong_count,
    ) = classify_ads_papers(
        papers
    )

    literature_score = min(
        sum(
            int(ref["evidence_score"])
            for ref in references
            if not ref["manual_review"]
        ),
        30,
    )

    return {
        "kic": kic,
        "query": query,
        "papers": papers,
        "references": references,
        "ads_n_papers": len(papers),
        "ads_relevant_papers": relevant_count,
        "ads_manual_review_papers": manual_count,
        "ads_strong_papers": strong_count,
        "literature_score": literature_score,
        "error": None,
        "from_cache": from_cache,
    }


# =============================================================================
# Benchmark levels
# =============================================================================

def determine_catalogue_level(
    vizier_score: int,
    strong_catalogue_count: int,
    asteroseismic_catalogue_count: int,
) -> int:
    """
    Catalogue-only evidence.

    This is deliberately separate from benchmark_level.
    """

    if strong_catalogue_count >= 2:
        return 3

    if strong_catalogue_count >= 1:
        return 2

    if asteroseismic_catalogue_count >= 2:
        return 1

    if vizier_score > 0:
        return 1

    return 0


def determine_benchmark_level(
    literature_score: int,
    strong_papers: int,
    vizier_score: int,
    strong_catalogue_count: int,
    asteroseismic_catalogue_count: int,
) -> int:
    """
    Combined benchmark level.

    Important:
    catalogue membership alone cannot produce level 3 or 4.

    Level 4:
        Multiple strong papers or very substantial literature evidence.

    Level 3:
        At least one strong star-specific paper.

    Level 2:
        Meaningful star-specific literature evidence, OR
        strong catalogue evidence without literature.

    Level 1:
        Catalogue candidate only / weak literature evidence.

    Level 0:
        No useful evidence.
    """

    if (
        strong_papers >= 2
        or literature_score >= 20
    ):
        return 4

    if (
        strong_papers >= 1
        or literature_score >= 12
    ):
        return 3

    if (
        literature_score >= 5
        or strong_catalogue_count >= 1
    ):
        return 2

    if (
        literature_score > 0
        or asteroseismic_catalogue_count >= 1
        or vizier_score > 0
    ):
        return 1

    return 0


# =============================================================================
# Build output
# =============================================================================

def extract_literature_numax_dnu(
    hits: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract νmax and Δν values from the validated VizieR catalogue rows.

    This is deliberately labelled as catalogue literature information,
    not independent ADS literature evidence.
    """

    numax_values = []
    dnu_values = []

    for hit in unique_vizier_hits(hits):

        row = hit.get(
            "catalogue_row",
            {},
        )

        lower_to_original = {
            str(k).lower(): k
            for k in row.keys()
        }

        numax_candidates = [
            "numax",
            "nu_max",
            "nu_max_lit",
            "numax_lit",
            "frequency_of_maximum_power",
        ]

        dnu_candidates = [
            "dnu",
            "delta_nu",
            "delnu",
            "delta_nu_lit",
        ]

        numax_column = None

        for candidate in numax_candidates:

            if candidate.lower() in lower_to_original:
                numax_column = lower_to_original[
                    candidate.lower()
                ]
                break

        dnu_column = None

        for candidate in dnu_candidates:

            if candidate.lower() in lower_to_original:
                dnu_column = lower_to_original[
                    candidate.lower()
                ]
                break

        if numax_column is not None:

            value = pd.to_numeric(
                row.get(numax_column),
                errors="coerce",
            )

            if pd.notna(value) and value > 0:
                numax_values.append(
                    float(value)
                )

        if dnu_column is not None:

            value = pd.to_numeric(
                row.get(dnu_column),
                errors="coerce",
            )

            if pd.notna(value) and value > 0:
                dnu_values.append(
                    float(value)
                )

    result = {}

    if numax_values:

        result["numax_lit_sources"] = len(
            numax_values
        )

        result["numax_lit_median"] = float(
            np.median(numax_values)
        )

        result["numax_lit_std"] = (
            float(np.std(numax_values, ddof=0))
            if len(numax_values) > 1
            else 0.0
        )

    else:

        result["numax_lit_sources"] = 0
        result["numax_lit_median"] = np.nan
        result["numax_lit_std"] = np.nan

    if dnu_values:

        result["dnu_lit_sources"] = len(
            dnu_values
        )

        result["dnu_lit_median"] = float(
            np.median(dnu_values)
        )

        result["dnu_lit_std"] = (
            float(np.std(dnu_values, ddof=0))
            if len(dnu_values) > 1
            else 0.0
        )

    else:

        result["dnu_lit_sources"] = 0
        result["dnu_lit_median"] = np.nan
        result["dnu_lit_std"] = np.nan

    return result


def build_catalogue_hits_output(
    hits_by_kic: Dict[str, List[Dict[str, Any]]],
) -> pd.DataFrame:

    rows = []

    for kic, hits in hits_by_kic.items():

        for hit in unique_vizier_hits(hits):

            row = {
                "kic": kic,
                "catalogue_id": hit[
                    "catalogue_id"
                ],
                "catalogue_name": hit[
                    "catalogue_name"
                ],
                "catalogue_kind": hit[
                    "catalogue_kind"
                ],
                "catalogue_weight": hit[
                    "catalogue_weight"
                ],
            }

            catalogue_row = hit.get(
                "catalogue_row",
                {},
            )

            for key, value in catalogue_row.items():

                if key in row:
                    key = (
                        "catalogue_"
                        + key
                    )

                row[key] = value

            rows.append(row)

    return pd.DataFrame(rows)


def build_reference_output(
    ads_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:

    rows = []

    for kic, result in ads_results.items():

        for reference in result.get(
            "references",
            [],
        ):

            row = dict(reference)

            row["kic"] = kic

            rows.append(row)

    return pd.DataFrame(rows)


def build_benchmark_output(
    df: pd.DataFrame,
    hits_by_kic: Dict[str, List[Dict[str, Any]]],
    ads_results: Dict[str, Dict[str, Any]],
    ads_candidates: Sequence[str],
) -> pd.DataFrame:

    ads_candidate_set = set(
        str(x)
        for x in ads_candidates
    )

    rows = []

    df_by_kic = (
        df.set_index("kic")
    )

    for kic in df["kic"].astype(str):

        star = df_by_kic.loc[kic]

        hits = hits_by_kic.get(
            kic,
            [],
        )

        unique_hits = unique_vizier_hits(
            hits
        )

        (
            vizier_score,
            strong_catalogue_count,
            asteroseismic_catalogue_count,
        ) = calculate_vizier_score(
            hits
        )

        catalogue_level = (
            determine_catalogue_level(
                vizier_score,
                strong_catalogue_count,
                asteroseismic_catalogue_count,
            )
        )

        literature_values = (
            extract_literature_numax_dnu(
                hits
            )
        )

        ads = ads_results.get(
            kic,
            {},
        )

        literature_score = int(
            ads.get(
                "literature_score",
                0,
            )
            or 0
        )

        strong_papers = int(
            ads.get(
                "ads_strong_papers",
                0,
            )
            or 0
        )

        benchmark_score = (
            vizier_score
            + literature_score
        )

        benchmark_level = (
            determine_benchmark_level(
                literature_score=literature_score,
                strong_papers=strong_papers,
                vizier_score=vizier_score,
                strong_catalogue_count=strong_catalogue_count,
                asteroseismic_catalogue_count=(
                    asteroseismic_catalogue_count
                ),
            )
        )

        simbad = ads.get(
            "simbad",
            {},
        )

        main_id = simbad.get(
            "main_id"
        )

        simbad_ids = simbad.get(
            "ids",
            [],
        )

        row = {
            "kic": kic,

            "main_id": main_id,

            "simbad_ids": (
                ";".join(
                    simbad_ids
                )
                if simbad_ids
                else ""
            ),

            "vizier_catalogue_count": len(
                unique_hits
            ),

            "vizier_catalogues": ";".join(
                hit["catalogue_name"]
                for hit in unique_hits
            ),

            "vizier_score": vizier_score,

            "catalogue_level": catalogue_level,

            "numax_lit_sources": (
                literature_values[
                    "numax_lit_sources"
                ]
            ),

            "numax_lit_median": (
                literature_values[
                    "numax_lit_median"
                ]
            ),

            "numax_lit_std": (
                literature_values[
                    "numax_lit_std"
                ]
            ),

            "dnu_lit_sources": (
                literature_values[
                    "dnu_lit_sources"
                ]
            ),

            "dnu_lit_median": (
                literature_values[
                    "dnu_lit_median"
                ]
            ),

            "dnu_lit_std": (
                literature_values[
                    "dnu_lit_std"
                ]
            ),

            "numax_pipeline_sources": star.get(
                "numax_pipeline_sources",
                0,
            ),

            "numax_pipeline_median": star.get(
                "numax_pipeline_median",
                np.nan,
            ),

            "numax_pipeline_std": star.get(
                "numax_pipeline_std",
                np.nan,
            ),

            "numax_pipeline_relative_std": star.get(
                "numax_pipeline_relative_std",
                np.nan,
            ),

            "dnu_pipeline_sources": star.get(
                "dnu_pipeline_sources",
                0,
            ),

            "dnu_pipeline_median": star.get(
                "dnu_pipeline_median",
                np.nan,
            ),

            "dnu_pipeline_std": star.get(
                "dnu_pipeline_std",
                np.nan,
            ),

            "dnu_pipeline_relative_std": star.get(
                "dnu_pipeline_relative_std",
                np.nan,
            ),

            "ads_candidate": int(
                kic in ads_candidate_set
            ),

            "ads_n_papers": int(
                ads.get(
                    "ads_n_papers",
                    0,
                )
                or 0
            ),

            "ads_relevant_papers": int(
                ads.get(
                    "ads_relevant_papers",
                    0,
                )
                or 0
            ),

            "ads_manual_review_papers": int(
                ads.get(
                    "ads_manual_review_papers",
                    0,
                )
                or 0
            ),

            "ads_strong_papers": strong_papers,

            "literature_score": literature_score,

            "benchmark_score": benchmark_score,

            "benchmark_level": benchmark_level,
        }

        # Add useful manifest metadata.
        for column in [
            "teff",
            "evolstate",
            "duration",
        ]:

            if column in star.index:
                row[column] = star[column]

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Main ADS processing
# =============================================================================

def run_ads_search(
    df: pd.DataFrame,
    hits_by_kic: Dict[str, List[Dict[str, Any]]],
    candidate_kics: Sequence[str],
    token: str,
    workers: int,
    refresh_cache: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Resolve SIMBAD and perform ADS searches for selected candidates.
    """

    results = {}

    candidate_kics = [
        str(kic)
        for kic in candidate_kics
    ]

    if not candidate_kics:
        return results

    print()
    print("=" * 80)
    print("SIMBAD + ADS CANDIDATE SEARCH")
    print("=" * 80)

    print(
        f"ADS candidates: {len(candidate_kics)}"
    )

    print(
        f"ADS workers:    {workers}"
    )

    print()

    # -------------------------------------------------------------------------
    # SIMBAD
    # -------------------------------------------------------------------------

    print(
        "[SIMBAD] Resolving identifiers "
        "for ADS candidates..."
    )

    for i, kic in enumerate(
        candidate_kics,
        start=1,
    ):

        simbad_data = query_simbad_one(
            kic=kic,
            refresh_cache=refresh_cache,
        )

        results[kic] = {
            "simbad": simbad_data,
        }

        if (
            i == 1
            or i % 100 == 0
            or i == len(candidate_kics)
        ):

            print(
                f"[SIMBAD] {i}/{len(candidate_kics)}"
            )

    print(
        "[SIMBAD] Candidate resolution finished."
    )

    # -------------------------------------------------------------------------
    # Prepare ADS arguments
    # -------------------------------------------------------------------------

    jobs = []

    for kic in candidate_kics:

        simbad_data = results[kic].get(
            "simbad",
            {},
        )

        main_id = simbad_data.get(
            "main_id"
        )

        simbad_ids = simbad_data.get(
            "ids",
            [],
        )

        vizier_ids = catalogue_identifiers(
            hits_by_kic.get(
                kic,
                [],
            )
        )

        aliases = unique_preserve_order(
            list(simbad_ids)
            + vizier_ids
        )

        # Do not feed the KIC itself repeatedly as an alias.
        aliases = [
            alias
            for alias in aliases
            if normalize_kic(alias) != kic
        ]

        jobs.append(
            (
                kic,
                main_id,
                aliases[:6],
            )
        )

    # -------------------------------------------------------------------------
    # ADS
    # -------------------------------------------------------------------------

    print()
    print(
        "[ADS] Searching candidates..."
    )

    start_time = time.time()

    completed = 0

    def worker(job):
        kic, main_id, aliases = job

        return search_ads_candidate(
            kic=kic,
            main_id=main_id,
            aliases=aliases,
            token=token,
            refresh_cache=refresh_cache,
        )

    with ThreadPoolExecutor(
        max_workers=max(1, workers)
    ) as executor:

        future_to_kic = {
            executor.submit(
                worker,
                job,
            ): job[0]
            for job in jobs
        }

        for future in as_completed(
            future_to_kic
        ):

            kic = future_to_kic[future]

            try:
                ads_result = future.result()

            except Exception as exc:

                ads_result = {
                    "kic": kic,
                    "query": "",
                    "papers": [],
                    "references": [],
                    "ads_n_papers": 0,
                    "ads_relevant_papers": 0,
                    "ads_manual_review_papers": 0,
                    "ads_strong_papers": 0,
                    "literature_score": 0,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    "from_cache": False,
                }

            # Preserve SIMBAD result.
            ads_result["simbad"] = results[kic].get(
                "simbad",
                {},
            )

            results[kic] = ads_result

            completed += 1

            if (
                completed == 1
                or completed % 50 == 0
                or completed == len(jobs)
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"[ADS] "
                    f"{completed}/{len(jobs)} "
                    f"({elapsed:.1f} s)"
                )

    print()
    print(
        "[ADS] Candidate search finished."
    )

    return results


# =============================================================================
# Summary
# =============================================================================

def print_summary(
    benchmark_df: pd.DataFrame,
    ads_candidate_count: int,
) -> None:

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"Total unique stars: "
        f"{len(benchmark_df)}"
    )

    print(
        f"ADS candidates: "
        f"{ads_candidate_count}"
    )

    print()

    print(
        "Catalogue-level distribution:"
    )

    if "catalogue_level" in benchmark_df:

        print(
            benchmark_df[
                "catalogue_level"
            ].value_counts()
            .sort_index()
            .to_string()
        )

    print()

    print(
        "Benchmark-level distribution:"
    )

    print(
        benchmark_df[
            "benchmark_level"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "Stars with ADS literature:"
    )

    print(
        int(
            (
                benchmark_df[
                    "ads_n_papers"
                ]
                > 0
            ).sum()
        )
    )

    print(
        "Stars with relevant literature:"
    )

    print(
        int(
            (
                benchmark_df[
                    "ads_relevant_papers"
                ]
                > 0
            ).sum()
        )
    )

    print(
        "Stars with strong literature:"
    )

    print(
        int(
            (
                benchmark_df[
                    "ads_strong_papers"
                ]
                > 0
            ).sum()
        )
    )

    print()

    print(
        "Top benchmark candidates:"
    )

    display_columns = [
        "kic",
        "main_id",
        "vizier_catalogue_count",
        "vizier_score",
        "numax_lit_sources",
        "numax_lit_std",
        "ads_n_papers",
        "ads_relevant_papers",
        "ads_strong_papers",
        "literature_score",
        "benchmark_score",
        "benchmark_level",
    ]

    display_columns = [
        c
        for c in display_columns
        if c in benchmark_df.columns
    ]

    top = benchmark_df.sort_values(
        [
            "benchmark_level",
            "benchmark_score",
            "vizier_score",
        ],
        ascending=False,
    ).head(20)

    print(
        top[display_columns].to_string(
            index=False
        )
    )


# =============================================================================
# Argument parser
# =============================================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Find well-studied Kepler benchmark stars "
            "using bulk VizieR + targeted SIMBAD/ADS searches."
        )
    )

    parser.add_argument(
        "--ads-candidates",
        type=int,
        default=1000,
        help=(
            "Number of stars to search in ADS. "
            "0 disables ADS; -1 searches all stars. "
            "Default: 1000."
        ),
    )

    parser.add_argument(
        "--ads-workers",
        type=int,
        default=4,
        help=(
            "Number of concurrent ADS requests. "
            "Default: 4."
        ),
    )

    parser.add_argument(
        "--max-stars",
        type=int,
        default=None,
        help=(
            "Optional limit on the number of unique KICs. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help=(
            "Ignore existing VizieR/SIMBAD/ADS caches "
            "and perform fresh queries."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    args = parse_arguments()

    ensure_directories()

    print()
    print("=" * 80)
    print("KEPLER BENCHMARK STAR SEARCH")
    print("=" * 80)

    print(
        f"Manifest:       {MANIFEST_PATH}"
    )

    print(
        f"ADS candidates: {args.ads_candidates}"
    )

    print(
        f"ADS workers:    {args.ads_workers}"
    )

    print(
        f"Refresh cache:  {args.refresh_cache}"
    )

    # -------------------------------------------------------------------------
    # Load manifest
    # -------------------------------------------------------------------------

    df = load_manifest()

    print()
    print(
        f"Loaded {len(df)} unique KICs."
    )

    if args.max_stars is not None:

        df = df.head(
            args.max_stars
        ).copy()

        print(
            f"Using first {len(df)} stars "
            f"because --max-stars was supplied."
        )

    df = add_pipeline_scatter(
        df
    )

    kics = df[
        "kic"
    ].astype(str).tolist()

    # -------------------------------------------------------------------------
    # VizieR bulk search
    # -------------------------------------------------------------------------

    hits_by_kic, vizier_status = (
        bulk_vizier_search(
            kics=kics,
            refresh_cache=args.refresh_cache,
        )
    )

    # -------------------------------------------------------------------------
    # Select ADS candidates
    # -------------------------------------------------------------------------

    candidate_kics = (
        select_ads_candidates(
            df=df,
            hits_by_kic=hits_by_kic,
            n_candidates=args.ads_candidates,
        )
    )

    print()
    print("=" * 80)
    print("ADS CANDIDATE SELECTION")
    print("=" * 80)

    print(
        f"Selected {len(candidate_kics)} "
        f"ADS candidates from {len(df)} stars."
    )

    if candidate_kics:

        print(
            "First candidates:"
        )

        for kic in candidate_kics[:20]:

            hits = hits_by_kic.get(
                kic,
                [],
            )

            score, strong, astro = (
                calculate_vizier_score(
                    hits
                )
            )

            print(
                f"  KIC {kic}: "
                f"VizieR score={score}, "
                f"strong_catalogues={strong}, "
                f"asteroseismic_catalogues={astro}"
            )

    # -------------------------------------------------------------------------
    # ADS + SIMBAD
    # -------------------------------------------------------------------------

    ads_results = {}

    if args.ads_candidates != 0:

        token = get_ads_token()

        if token is None:

            print()
            print(
                "WARNING: ADS_API_TOKEN is not set."
            )

            print(
                "Skipping ADS/SIMBAD literature search."
            )

            print(
                "Set it with:"
            )

            print(
                '  export ADS_API_TOKEN="your-token"'
            )

        else:

            if test_ads(token):

                ads_results = run_ads_search(
                    df=df,
                    hits_by_kic=hits_by_kic,
                    candidate_kics=candidate_kics,
                    token=token,
                    workers=args.ads_workers,
                    refresh_cache=args.refresh_cache,
                )

            else:

                print(
                    "ADS test failed; "
                    "skipping ADS searches."
                )

    # -------------------------------------------------------------------------
    # Build outputs
    # -------------------------------------------------------------------------

    benchmark_df = build_benchmark_output(
        df=df,
        hits_by_kic=hits_by_kic,
        ads_results=ads_results,
        ads_candidates=candidate_kics,
    )

    catalogue_df = (
        build_catalogue_hits_output(
            hits_by_kic
        )
    )

    references_df = (
        build_reference_output(
            ads_results
        )
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    benchmark_df.to_csv(
        STARS_OUTPUT,
        index=False,
    )

    references_df.to_csv(
        REFERENCES_OUTPUT,
        index=False,
    )

    catalogue_df.to_csv(
        CATALOGUE_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    print_summary(
        benchmark_df=benchmark_df,
        ads_candidate_count=len(
            candidate_kics
        ),
    )

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        f"Stars written to:       {STARS_OUTPUT}"
    )

    print(
        f"References written to:  {REFERENCES_OUTPUT}"
    )

    print(
        f"VizieR hits written to: {CATALOGUE_OUTPUT}"
    )

    print(
        f"Cache directory:        {CACHE_DIR}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
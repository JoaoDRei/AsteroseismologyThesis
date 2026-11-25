# KEPSEISMIC granulation analysis pipeline

This is deliberately separate from the ML preprocessing pipeline. It does not
standardize each light curve or PSD.

## Directory layout

```text
Granulation/
├── process_lightcurves.py
├── fit_granulation.py
├── analyze_granulation_fits.py
├── requirements.txt
├── raw_kepseismic/              # downloaded HLSP/FITS tree
└── ...
```

The fitter's input and output data directories are under the thesis project
root (two levels above `Granulation/`), not inside this directory.

## Environment

Use a dedicated environment or your existing thesis environment:

```bash
pip install -r requirements.txt
```

## 1. Process FITS light curves

Put the downloaded KEPSEISMIC folder under `raw_kepseismic/`, or pass its path:

```bash
python process_lightcurves.py \
  --raw-root ./raw_kepseismic \
  --output ./data/psd_physical \
  --min-frequency 0.1 \
  --max-frequency 283
```

The script searches recursively for `*cor-filt-inp.fits`. It subtracts the
median but does not divide by the standard deviation. Sigma clipping is off by
default. Do not enable it for the production run until its effect has been
inspected.

**Frequency units:** the script converts the Astropy frequency Quantity to
microhertz. This is important: Lightkurve Kepler frequencies are commonly
represented in cycles/day. A numerical value of 283 in cycles/day is NOT
283 microhertz. The old ML preprocessing code should be audited for this
unit conversion before relying on its `freq <= 283` mask as a 283-uHz cutoff.

Each PSD is stored as an NPZ with `frequency_uHz` and `power`, plus a JSON
sidecar recording units and processing choices.

## 2. Fit the background

Edit the settings at the top of `fit_granulation.py` before running:

- `MAX_STARS`: number of distinct KICs to fit; `0` means all KICs. All PSD
  files available for each selected KIC are included.
- `MIN_FREQUENCY_UHZ` and `MAX_FREQUENCY_UHZ`: frequency range to fit.
- `NU_MAX_CSV`: optional CSV path with columns `KIC,nu_max` (microhertz), or
  `None` to omit the known oscillation-envelope centres.

The input and output paths are based on the thesis project root, not the
current working directory. Run without command-line options:

```bash
python fit_granulation.py
```

The fit parameters are saved to `data/fitted_parameters/fit_results.csv`,
failures to `data/fitted_parameters/fit_failures.csv`, and one diagnostic plot
per PSD file to `results/plots/`.

The fitter uses two Harvey-like components, a white-noise term, and a Gaussian
oscillation-envelope nuisance term. It reports characteristic frequencies and
timescales, with `tau = 1/(2*pi*nu_char)`. It also reports component heights
in the native PSD power units.

## 3. Run the catalogue-constrained v2 fitter

The v2 fitter uses `Baseline/manifest.csv` to associate physical PSDs with
catalogue `nu_max` values. Run it from this directory without arguments:

```bash
python fit_granulation_v2.py
```

Its paths are based on the thesis project root, not the current working
directory. Set the `MAX_STARS` constant near the top of
`fit_granulation_v2.py` to choose how many KICs to fit (default: `0`, meaning
all). The script also has a `FIT_ALPHA` setting (default: `True`) to fit
the Harvey slopes within [2, 6]; set it to `False` to fix both slopes at 4.
For a limited run, KICs are selected randomly but reproducibly using the
`SAMPLE_SEED` setting (default: 42). Each KIC contributes up to three
duration-specific PSD fits.
Results and plots go to model- and sample-specific directories. The default
full free-slope run writes to `data/fitted_parameters_v2_all_free_alpha/` and
`results/granulation_v2_plots_all_free_alpha/`, preserving the previous
full-run results. A 150-KIC free-slope run uses names ending in
`_150_free_alpha`. Both Harvey slopes are fixed at 4 when `FIT_ALPHA` is
`False`; `--fit-alpha` and `--fixed-alpha` override the setting. Plots are
saved by default; use `--no-plot` to skip them. The current manifest and
physical PSD set contain 9,939 KICs (29,809 spectra). It is a
catalogue-constrained variant, so its results are not directly interchangeable
with the unconstrained v1 results.

The fit objective is the RMS difference in log10 power between the fitted
model and log-binned PSD medians. A typical value around 0.16 dex is not by
itself evidence of a failed optimizer; inspect the residual panel and compare
the fit against v1 or a more flexible model. The catalogue-constrained model
fixes the oscillation-envelope centre and bounds free Harvey slopes, so some
stars will be poorly represented even when optimization succeeds. A 150-KIC
reproducible random-sample trial with free slopes reduced median log-RMS from
about 0.161 to 0.151 dex; however, the long-component slope hit a bound in
186 of 450 fits. Inspect these boundary cases before interpreting the slopes
physically. Diagnostic plots include a log-power fit and a residual panel;
very poor individual fits should be reviewed rather than hidden by the overall
median.

## 4. Analyze fit quality

Run the analyzer without arguments to analyze the full-sample v2 fit:

```bash
python analyze_granulation_fits.py
```

It reads `data/fitted_parameters_v2_all_free_alpha/fit_results.csv` and writes
reports, tables, and plots to `results/granulation_v2_qc_all_free_alpha/`.
The analyzer reads the adjacent `run_config.json` to use the fit's alpha and
envelope-width bounds for boundary flags and histogram guides. You can still
analyze v1 by passing its CSV:

```bash
python analyze_granulation_fits.py \
  ../../data/fitted_parameters/fit_results.csv \
  --output-dir ../../results/granulation_qc
```

For alternate/custom CSVs, v2 is detected from its `nu_max_catalogue` column
when no adjacent run configuration is available. Bounds can be overridden
with `--alpha-bounds` and `--envelope-width-bounds`. The analyzer accepts both
result schemas and derives duration from each input filename.

## 5. Compare v1 and v2 fits

After both fitters have produced results, compare their matched
KIC-duration spectra:

```bash
python compare_granulation_fits.py
```

The comparison defaults to the v1 and v2 CSVs above, keeps only successful
fits present in both, and writes matched rows, metric summaries,
duration-stability summaries, and scatter plots to
`results/granulation_comparison_all_free_alpha/`. It compares the new full
free-slope run against the original v1 results; no analyzer output is needed
as input. The quality analyzer is optional and can be run before or after
comparison.

**Scientific limitation:** these component heights are exploratory fit
parameters. They are not automatically the same definition as Corsaro et al.'s
mesogranulation amplitude. Before comparing exponents or normalizations with
that paper, reproduce its exact PSD normalization, background model, component
identification, and amplitude definition. The current log-binned least-squares
fit is a practical first-pass fitter, not a final inference pipeline. It does
not yet propagate full PSD likelihood uncertainties.

## 6. Join catalogue and explore metallicity

Create a CSV with at least `KIC,metallicity`. If available, include
`nu_max`, `delta_nu`, `teff`, and a documented `mass` column.

```bash
python analyse_results.py \
  --fits ./data/fitted_parameters/fit_results.csv \
  --catalogue ./apokasc_join.csv \
  --metallicity-column metallicity \
  --output ./results
```

The regression is exploratory. Check sample selection, parameter domains,
uncertainties, evolutionary state, and correlations before drawing conclusions.
Do not use ML-predicted metallicity for the initial physical test; use the
catalogue spectroscopic metallicities.

## Important checks before the production run

1. Inspect the FITS headers and confirm the flux unit and meaning of
   `cor-filt-inp` for the exact KEPSEISMIC release used.
2. Compare raw/corrected light curves and their PSDs for several stars.
3. Confirm the Lomb–Scargle PSD normalization and units with a synthetic
   signal or a variance/integrated-power check.
4. Confirm the actual long-cadence Nyquist frequency in the data. 283 uHz is
   the usual approximate Kepler long-cadence Nyquist frequency.
5. Inspect fit plots for a range of nu_max, metallicity, and evolutionary
   states. Reject fits with poor residuals or parameters pinned to bounds.
6. Do not interpret either Harvey component as mesogranulation solely from
   its ordering. Establish the correspondence to the literature model.

## Suggested HPC execution

Start interactively with `--limit 20`. For large runs, use the cluster's
standard batch scheduler and set BLAS/OpenMP thread counts conservatively,
for example `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. The scripts are
single-process; parallelize later by splitting the PSD file list into batches.

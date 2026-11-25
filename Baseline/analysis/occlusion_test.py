#!/usr/bin/env python3
"""Simple frequency-band occlusion for an existing PSD model.

Run:
 python -m Baseline.analysis.occlusion_test

The default run is the PSD ResNet1D multitask experiment, using the ten stars
from the IG robustness pilot. Results are saved both in the run's
occlusion_test/ directory and in Baseline/results_analysis/occlusion_test/.
These defaults can be overridden with command-line options.

To select other stars:
 python -m Baseline.analysis.occlusion_test --kics 11245339,11805876
or:
 python -m Baseline.analysis.occlusion_test --max-stars 50 --seed 42

Uses the same AstroBaselineDataset and model factory as interpret.py.
No retraining; labels are not needed for the output-change analysis.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from Baseline.training.interpret import (
    build_model, load_data, normalize_kic, safe_float,
)
from Baseline.paths import BASELINE_DIR


DEFAULT_RUN_DIR = (
    BASELINE_DIR
    / 'results'
    / 'resnet1d_psd_80d_nu_max_delta_nu_class_20261018_190922_635749'
)
DEFAULT_KICS = (
    '11245339,11805876,12784948,1429505,3533464,'
    '4348593,4991033,5172229,6501869,9021799'
)
DEFAULT_ANALYSIS_OUTPUT_DIR = (
    BASELINE_DIR / 'results_analysis' / 'occlusion_test'
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument(
        '--output-dir',
        type=Path,
        help='Primary output directory; defaults to <run-dir>/occlusion_test',
    )
    p.add_argument(
        '--kics',
        type=str,
        default=DEFAULT_KICS,
        help='Comma-separated KICs; defaults to the IG robustness pilot stars',
    )
    p.add_argument('--max-stars', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.max_stars < 1:
        p.error('--max-stars must be positive')

    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / 'config.yaml'
    checkpoint_path = run_dir / 'best_model.pth'
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError('Expected config.yaml and best_model.pth in --run-dir')
    with config_path.open(encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config['data']['mode'] != 'psd':
        raise ValueError('This experiment only supports PSD input.')
    if config.get('debug', {}).get('synthetic', False):
        raise ValueError('This experiment requires real PSDs and catalogue nu_max.')

    requested = ([v.strip() for v in args.kics.split(',') if v.strip()]
                 if args.kics is not None else None)
    if args.kics is not None and not requested:
        p.error('--kics cannot be empty')
    dataset, _, df = load_data(config, str(config_path), kics=requested)
    # Select unique KICs, avoiding repeated durations and accidental duplicates.
    kic_strings = df['kic'].map(normalize_kic)
    if requested is None:
        unique = kic_strings.drop_duplicates().to_numpy()
        rng = np.random.default_rng(args.seed)
        selected = set(rng.choice(unique, size=min(args.max_stars, len(unique)), replace=False))
    else:
        selected = set(map(normalize_kic, requested))
    selected_indices = []
    seen = set()
    for i, kic in enumerate(kic_strings):
        if kic in selected and kic not in seen:
            selected_indices.append(i)
            seen.add(kic)
    if not selected_indices:
        raise ValueError('No stars selected.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    example, _, _ = dataset[selected_indices[0]]
    model = build_model(config, example.shape[-1]).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        elif 'state_dict' in state:
            state = state['state_dict']
    model.load_state_dict(state)
    model.eval()

    targets = config['model']['targets']
    output_dir = (
        args.output_dir or (run_dir / 'occlusion_test')
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for index in selected_indices:
            row = df.iloc[index]
            x, _, kic = dataset[index]
            kic = normalize_kic(kic)
            nu = safe_float(row.get('nu_max'))
            if nu is None or nu <= 0:
                print(f'Skipping KIC {kic}: missing positive catalogue nu_max')
                continue
            # The model PSD uses a uniform log-frequency grid. Reconstruct the
            # actual grid from the saved source PSD, not an assumed fixed range.
            source_obj = np.load(row['psd_path'])
            try:
                source = (source_obj[source_obj.files[0]] if isinstance(source_obj, np.lib.npyio.NpzFile)
                          else source_obj)
                freq = np.asarray(source[0])
            finally:
                if isinstance(source_obj, np.lib.npyio.NpzFile):
                    source_obj.close()
            freq = freq[(freq >= 0.1) & (freq <= 283)]
            freq = np.sort(freq)
            if len(freq) < 10 or not np.all(np.isfinite(freq)) or np.any(freq <= 0):
                raise ValueError(f'Invalid frequency axis for KIC {kic}')
            axis = 10 ** np.linspace(np.log10(freq[0] + 1e-8),
                                    np.log10(freq[-1] + 1e-8), x.shape[-1])
            x = x.unsqueeze(0).to(device)
            original = model(x).reshape(-1).cpu().numpy()
            if len(original) != len(targets):
                raise ValueError('Model output does not match config targets')
            masks = {
                'hump': (axis >= 0.5 * nu) & (axis <= 1.5 * nu),
                'low': axis < 0.5 * nu,
            }
            for band, mask in masks.items():
                if not mask.any():
                    print(f'Skipping empty mask {band} for KIC {kic}')
                    continue
                modified = x.clone()
                modified[..., torch.from_numpy(mask).to(device)] = 0.0
                changed = model(modified).reshape(-1).cpu().numpy()
                for j, target in enumerate(targets):
                    records.append({
                        'kic': kic, 'target': target, 'band': band,
                        'nu_max_catalog_uHz': nu,
                        'n_bins_masked': int(mask.sum()),
                        'fraction_bins_masked': float(mask.mean()),
                        'original_output': float(original[j]),
                        'occluded_output': float(changed[j]),
                        'delta_output': float(changed[j] - original[j]),
                        'abs_delta_output': float(abs(changed[j] - original[j])),
                    })
            print(f'KIC {kic}: complete')
    if not records:
        raise RuntimeError('No valid stars with usable masks.')
    result = pd.DataFrame(records)
    result.to_csv(output_dir / 'occlusion_per_star.csv', index=False)
    # Normalize output changes by the variability of *unoccluded* model outputs
    # across the same selected stars, separately for each target.
    original_per_target = result.drop_duplicates(['kic', 'target'])
    scales = original_per_target.groupby('target')['original_output'].std(ddof=0)
    result['reference_output_std'] = result['target'].map(scales)
    result['abs_delta_in_output_std'] = np.where(
        result['reference_output_std'] > 0,
        result['abs_delta_output'] / result['reference_output_std'],
        np.nan,
    )
    result.to_csv(output_dir / 'occlusion_per_star.csv', index=False)
    summary = (result.groupby(['target', 'band'], as_index=False)
               .agg(n_stars=('kic', 'nunique'),
                    median_fraction_bins_masked=('fraction_bins_masked', 'median'),
                    median_abs_delta_output=('abs_delta_output', 'median'),
                    median_abs_delta_in_output_std=('abs_delta_in_output_std', 'median'),
                    mean_abs_delta_in_output_std=('abs_delta_in_output_std', 'mean')))
    summary.to_csv(output_dir / 'occlusion_summary.csv', index=False)
    (output_dir / 'experiment_settings.json').write_text(json.dumps({
        'run_dir': str(run_dir), 'checkpoint': str(checkpoint_path),
        'selected_kics': [normalize_kic(df.iloc[i]['kic']) for i in selected_indices],
        'seed': args.seed, 'mask_replacement': 'zero in standardized log-PSD',
        'hump': '[0.5,1.5] * catalogue nu_max',
        'low': 'frequency < 0.5 * catalogue nu_max',
        'output_scale': 'raw model output (possibly log-transformed)',
    }, indent=2), encoding='utf-8')
    analysis_output_dir = DEFAULT_ANALYSIS_OUTPUT_DIR.resolve()
    if analysis_output_dir != output_dir:
        analysis_output_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            'occlusion_per_star.csv',
            'occlusion_summary.csv',
            'experiment_settings.json',
        ):
            shutil.copy2(
                output_dir / filename,
                analysis_output_dir / filename,
            )
    print('\n', summary.to_string(index=False))
    print(f'\nSaved run results to {output_dir}')
    if analysis_output_dir != output_dir:
        print(f'Mirrored analysis results to {analysis_output_dir}')


if __name__ == '__main__':
    main()

# Main paper figures

Place `figure_scripts/` at the root of EECS182-Project (alongside `splits.json`). Run from any working directory. The default output directory is the repository's `figures/` directory.

```bash
python -m pip install numpy pandas matplotlib pillow
for n in 1 2 3 4 5; do python figure_scripts/plot_figure${n}.py; done
```

Figure 1 checks the repository README, splits and final setting. Copy the original raster assets into `figure_scripts/assets/`:
- `brain_setup_cropped.png`
- `figure1_on_traces_crop.png`

Figures 2–5 use the original analysis-generated source tables, not substitute scores from other experiments. Copy these files from the original figure output directories into `figure_scripts/source_data/` (or specify `--source-dir` separately for each script):

| Figure | Files |
|---|---|
| 2 | `lfp_traces.csv`, `saved_spectra.csv`, `signal_examples.csv`, `ridge_pairs.csv` |
| 3 | `matched_per_draw.csv` |
| 4 | `trace_source_data.csv`, `control_per_session.csv` |
| 5 | `figure5_curve_per_seed.csv` |

These files are not included in this package and were not found in the repository directories inspected. They were written by the original uploaded scripts. Retain the original scripts and analysis outputs privately for upstream provenance. Check lab data-sharing approval before committing any source data, particularly LFP and velocity traces or session-level data.

The revised scripts retain the original rendering and day-bootstrap functions. They do not redo model inference, checkpoint checks, or query-mask verification; those checks belong to the upstream analysis that generated the source tables. No local Python helper modules or model checkpoints are required for rendering. Python syntax and CLI startup were checked; full rendering with the actual source tables and assets could not be tested because those files were not supplied.

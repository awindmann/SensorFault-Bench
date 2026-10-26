# Figure Setup

This optional folder renders the figures of the NeurIPS 2026 paper
"Benchmarking Sensor-Fault Robustness in Forecasting" from fixed CSV inputs,
independently of the Python benchmark workflow.

## Folder layout

The figure scripts assume this layout under `paper_artifact/R/`:

- `code/`: figure source files
- `data/`: input CSVs and curated trace folders
- `figures/`: generated outputs
- `library/`: optional local R package library

The scripts look for packages in `library/` first and then in the usual R
library paths. Because the repository does not ship `library/`, the install
command below recreates it.

## R packages

The figure scripts use these packages:

- `ggplot2`
- `dplyr`
- `ggrepel`
- `ggh4x`
- `tikzDevice`
- `scales`
- `patchwork`

Install them from the repository root with:

```bash
mkdir -p paper_artifact/R/library
Rscript -e 'lib <- "paper_artifact/R/library"; dir.create(lib, recursive = TRUE, showWarnings = FALSE); .libPaths(c(normalizePath(lib), .libPaths())); install.packages(c("ggplot2","dplyr","ggrepel","ggh4x","tikzDevice","scales","patchwork"), repos = "https://cloud.r-project.org")'
```

## LaTeX / TikZ tools

Several figures export through `tikzDevice` and therefore need a working TeX
installation with these command-line tools:

- `pdflatex`
- `lualatex`
- `pdfcrop` for Figure 2
- `gs` (Ghostscript, which `pdfcrop` also needs) for the bounding box of
  Figure 2

On Debian or Ubuntu, start with:

```bash
sudo apt-get update
sudo apt-get install -y \
  texlive-latex-base \
  texlive-latex-recommended \
  texlive-luatex \
  texlive-pictures \
  texlive-fonts-recommended \
  texlive-extra-utils
```

Notes:

- Figures 2, 5, and 6 compile with `pdflatex`.
- Figure 3 exports through TikZ and compiles with `pdflatex`.
- Figure 4 exports through TikZ and compiles the final `.tex` file with
  `lualatex`.
- Figure 4 sets up a writable LuaTeX cache at runtime, so the cache needs no
  manual setup.
- Because its TikZ export is large, Figure 4 takes longer to compile than the
  other figures.

## `pdfcrop` on macOS and Windows

The common TeX distributions include `pdfcrop` on macOS and Windows as well as
on Linux:

- macOS: MacTeX
- Windows: TeX Live or MiKTeX

Check the installation with:

```bash
pdfcrop --version
```

If the command is missing:

- install a full TeX distribution
- make sure the TeX binaries are on `PATH`
- on Windows, reopen the terminal after installation so the updated `PATH` is
  picked up

## Input data locations

The figures read these inputs:

- Figure 2: `data/backbone_scenario_heatmap_data.csv`
- Figure 3: `data/forecast_plots_final_traces/...`
- Figure 4: `data/forecast_plots_final_traces/...`
- Figure 5: `data/main_table_backbone.csv`
- Figure 6: `data/pgd_trajectory_data.csv`

The forecast trace CSVs contain only the forecast windows shown in Figures 3
and 4. They serve as plotting inputs, so the larger trace outputs per sample of
the analysis run are left out. Because these traces store no
`robustness_method` column, replaying them with
`scripts/render_forecast_plots.py from-traces` needs
`--missing-robustness-method baseline`.

## Running the figures

Run the scripts from the repository root:

```bash
Rscript paper_artifact/R/code/Figure_2_backbone_scenario_heatmap.R
Rscript paper_artifact/R/code/Figure_3_forecast_short_term.R
Rscript paper_artifact/R/code/Figure_4_forecast_long_term.R
Rscript paper_artifact/R/code/Figure_5_trade_off_views.R
Rscript paper_artifact/R/code/Figure_6_trajectories.R
```

Outputs are written to `paper_artifact/R/figures/`.

## Expected outputs

- `Figure_2_backbone_scenario_heatmap.pdf`
- `Figure_3_forecast_short_term.pdf`
- `Figure_4_forecast_long_term.pdf`
- `Figure_5_backbone_pareto_by_dataset.pdf`
- `Figure_5_clean_vs_worst_error_by_dataset.pdf`
- `Figure_6_trajectories.pdf`

## Verification

`paper_artifact/MANIFEST.json` records the SHA-256 checksum and byte size of
every file in this folder.

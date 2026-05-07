# Figure Setup

This optional subartifact renders the polished paper figures from frozen CSV
inputs. It is separate from the Python benchmark workflow.

## Folder layout

The figure scripts assume this layout under `paper_artifact/R/`:

- `code/`: figure source files
- `data/`: input CSVs and curated trace folders
- `figures/`: generated outputs
- `library/`: optional local R package library

The scripts automatically look for packages in `library/` first, then fall
back to the normal R library paths. `library/` is a local convenience cache and
is not distributed. Other users can recreate the package environment with the
install command below.

## R packages

Install the union of packages used by the figure scripts:

- `ggplot2`
- `dplyr`
- `ggrepel`
- `ggh4x`
- `tikzDevice`
- `scales`
- `patchwork`

Recommended install command from the repository root:

```bash
mkdir -p paper_artifact/R/library
Rscript -e 'lib <- "paper_artifact/R/library"; dir.create(lib, recursive = TRUE, showWarnings = FALSE); .libPaths(c(normalizePath(lib), .libPaths())); install.packages(c("ggplot2","dplyr","ggrepel","ggh4x","tikzDevice","scales","patchwork"), repos = "https://cloud.r-project.org")'
```

## LaTeX / TikZ tools

Several figures export through `tikzDevice`, so a working TeX installation is required.

Required command-line tools:

- `pdflatex`
- `lualatex`
- `pdfcrop` for Figure 2

On Debian or Ubuntu, this is a reasonable starting point:

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
- Figure 3 now exports through TikZ and compiles with `pdflatex`.
- Figure 4 now exports through TikZ and compiles the final `.tex` with `lualatex`.
- Figure 4 sets a writable LuaTeX cache automatically at runtime, so no manual cache setup should be needed.
- Figure 4 can take noticeably longer to compile than the other figures because its TikZ export is large.

## `pdfcrop` on macOS and Windows

`pdfcrop` is not Linux-only. It is also available on macOS and Windows as part of common TeX distributions.

- `macOS`: typically available after installing MacTeX
- `Windows`: typically available after installing TeX Live or MiKTeX

Quick check:

```bash
pdfcrop --version
```

If the command is missing, the usual fixes are:

- install a full TeX distribution
- make sure the TeX binaries are on `PATH`
- on Windows, reopen the terminal after installation so the updated `PATH` is picked up

## Input data locations

Current expected inputs:

- Figure 2: `data/backbone_scenario_heatmap_data.csv`
- Figure 3: `data/forecast_plots_final_traces/...`
- Figure 4: `data/forecast_plots_final_traces/...`
- Figure 5: `data/main_table_backbone.csv`
- Figure 6: `data/pgd_trajectory_data.csv`

The forecast trace CSVs include only the forecast windows shown in Figures 3 and 4.
They are plotting inputs, not full evaluation replay files, and the larger
per-sample trace outputs from the analysis run are intentionally omitted.
These bundled trace extracts predate the current replay trace schema and do not
store a `robustness_method` column. When replaying them with
`scripts/render_forecast_plots.py from-traces`, pass
`--missing-robustness-method baseline` explicitly.

## Running the figures

Run from the repo root:

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

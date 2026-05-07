# Paper Artifact

This directory is the frozen paper-artifact identity for the NeurIPS 2026
submission. It contains a manifest-indexed result table bundle and a figure
reproduction surface.

## Bundle Contents

- `MANIFEST.json`: machine-readable index for the result table bundle,
  including checksums, dataset scope, source identities, generation commands,
  and intentional omissions.
- `tables/`: CSV analysis tables organized by bundled evidence source, including
  the primary clean-selected results, selector-pressure sensitivity tables,
  evaluation-data seed sensitivity tables, and fixed selected-channel-fraction
  sensitivity tables.
- `paper_inputs/`: derived paper-facing CSV inputs generated from the bundled
  primary tables.
- `config/eval_context.json`: public-safe evaluation context for the bundled
  analysis sources.
- `forecast_examples/manifest.json`: curated qualitative forecast-example sample
  identities and trace-bundle status.
- `R/`: figure reproduction surface with paper figure scripts, figure input
  CSVs, curated forecast traces, and polished figure PDFs.

The artifact intentionally excludes checkpoints, MLflow run stores, raw
datasets, processed runtime datasets, broad sample-level dumps, raw private
analysis configs, full forecast trace dumps outside the curated R extracts, and
complete historical export trees. Dataset metadata for the two curated derived
records is handled separately through the dataset submission route.

## Paper Inputs

The files in `paper_inputs/` are the small CSV inputs used by the paper's table
and figure workflows. They are generated from the bundled primary source tables
during artifact construction and are listed in `MANIFEST.json` with row counts,
checksums, and source-table provenance.

## Provenance

Use paths relative to `paper_artifact/` when inspecting the bundle.
`MANIFEST.json` records checksums, source labels, and the original export paths
used to build the bundled tables.

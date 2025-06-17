# Tornado Viewer
Visual display for tornado plots - Fred Hutch Bioinformatics Shared Resource

## Input Data

Each dataset which can be parsed by this app contains:

- One or more binned sequencing depth files (CSV)
- One metadata table (CSV) with columns for chrom, start, end, peak_id, sample_groups, peak_no, and peak_group

All of those CSV files have the same number of rows.

## Development

Set up your development environment:

```
uv init
uv add cirro pandas plotly marimo matplotlib
```

Launch the app in editable notebook format:

```
bash edit.sh
```

Launch the app locally via HTML-WASM

```
bash build.sh
```

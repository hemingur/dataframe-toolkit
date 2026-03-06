# dfstat — DataFrame analysis and manipulation toolkit

`dfstat` is a command-line toolkit for exploratory data analysis on TSV files.
Each subcommand reads tabular data, performs one operation, and writes TSV to
stdout, making it easy to chain commands in pipelines.

## Installation

```bash
uv sync                      # install into project venv
source .venv/bin/activate    # or: source ~/venv/current/bin/activate
dfstat --help
```

## Quick example

```bash
dfstat stat data.tsv -c height weight -g sex
dfstat stat data.tsv -c value -g group -o \
  | dfstat pivot -i group -v value_mean -o \
  | dfstat print
```

## Subcommands

### Data transformation

| Command  | Description                                                                          |
|----------|--------------------------------------------------------------------------------------|
| `eval`   | Add or modify columns: eval expressions, string/path functions, statistical ops      |
| `query`  | Filter rows with pandas query expressions or SQL (`--sql` via DuckDB)                |
| `merge`  | Join two tables on key columns (inner/left/right/outer)                              |
| `melt`   | Reshape wide-to-long (`pd.melt`)                                                     |
| `pivot`  | Reshape long-to-wide with per-cell aggregation                                       |
| `func`   | Column transforms: cumsum, group mean/sum/min/max/count/median/std, rank, qcut:N     |
| `scale`  | Normalise: z-score, min-shift, sum/max/mean scaling, Blom rank, regression residuals |
| `interp` | Interpolate values from a reference curve into a table (1-D lookup)                  |

### Statistics

| Command | Description                                                                              |
|---------|------------------------------------------------------------------------------------------|
| `stat`  | Descriptive statistics (count/mean/std/CI/SEM/skew/kurtosis) with grouping and bootstrap |
| `fit`   | OLS/robust/weighted regression via R-style formulas; tidy table, `--summary`, `--anova`  |

### Plots

All plot commands write a PNG/PDF when `-f FILE` is given, or display
interactively otherwise. They support `--groupcol` for colour grouping,
`--subgraphcol` for subplot grids, and figure/font presets for
publication-quality output.

| Command | Description                                                                                     |
|---------|-------------------------------------------------------------------------------------------------|
| `scat`  | Scatter plot; optional OLS/robust fit overlay, bubble size (`--sizecol`), colour (`--colorcol`) |
| `line`  | Line plot; optional error bars (`--yerr`) or CI bands (`--yci lo,hi`) and fit overlay           |
| `hist`  | Histogram or KDE (`--kde`); normalisation, cumulative mode, mean±σ annotation (`--stats`)       |

### Utilities

| Command | Description                                                              |
|---------|--------------------------------------------------------------------------|
| `print` | Read any dfstat input (TSV, stdin, parquet pipe) and write TSV to stdout |
| `clean` | Remove leftover temp parquet pipe files from interrupted pipelines       |
| `help`  | List all subcommands or show full help for one: `dfstat help stat`       |

## Common patterns

### Chaining commands

```bash
# z-score within group, then fit a model
dfstat scale data.tsv -c expr -g condition -o \
  | dfstat fit - -f "expr_z ~ time + batch" -g condition
```

### Group summary then plot

```bash
dfstat stat results.tsv -c value -g group -o \
  | dfstat line - -x group -y value_mean --yerr value_sem -f fig.png
```

### Wide-to-long then plot overlaid histograms

```bash
dfstat melt data.tsv -i sample -d gene -v expression \
  | dfstat hist - -x expression -g gene -k -f dist.png
```

### Interpolation (standard curve lookup)

```bash
dfstat interp samples.tsv --ref stdcurve.tsv \
    -x fluorescence -v concentration -d conc_ng_ul
```

### Bootstrap confidence intervals

```bash
dfstat stat data.tsv -c value -g group --bootstrap 1000 --randomseed 42 -o \
  | dfstat pivot -i group -v value_mean -f mean -o \
  | dfstat stat - -c group_A group_B
```

## Input/output

All commands accept:

- A TSV filename as a positional argument
- `-` to read from stdin (TSV)
- A `.parquet` path written by a previous `dfstat` command with `-o`

Standard output options (available on all tabular commands):

- `-o` / `--output` — write a temp parquet for the next piped command instead of TSV to stdout
- `--select col1 col2 …` — keep only these columns
- `--drop col1 col2 …` — remove these columns
- `--round N` / `--sigdig N` — round numeric output
- `--postquery EXPR` — filter output rows after processing

## Figure options (plot commands)

```text
--size single|double|full|WxH    figure size (single ≈ 3.5", double ≈ 7.2")
--fontsize screen|publication|presentation
-f FILE                          save to file (PNG/PDF/SVG); omit to display
--groupcol COL                   colour-code by this column
--subgraphcol COL                split into subplot grid by this column
--ncols N                        columns in subplot grid (default: auto)
--legend TEXT                    legend label for ungrouped series
```

## Dependencies

- `pandas`, `numpy`, `scipy` — core data handling and statistics
- `statsmodels` — regression (`fit`, `scale --resid`)
- `duckdb` — SQL queries (`query --sql`)
- `matplotlib`, `seaborn` — plots
- `polars`, `pyarrow` — parquet I/O backend

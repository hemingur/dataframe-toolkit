# dftk — DataFrame analysis and manipulation toolkit

`dftk` is a command-line toolkit for exploratory data analysis on TSV files.
Each subcommand reads tabular data, performs one operation, and writes TSV to
stdout, making it easy to chain commands in pipelines.

`dftk` was developed during computational genomics research as a fast,
composable alternative to writing one-off pandas scripts. It is designed
for analysts who live in the terminal and want a consistent, pipeable toolkit
for the full data analysis pipeline — from initial exploration through
statistical modelling and publication-quality figures.

## Installation

```bash
uv tool install git+https://github.com/hemingur/dataframe-toolkit.git
dftk --help
```

To install from a local clone:

```bash
git clone https://github.com/hemingur/dataframe-toolkit.git
cd dataframe-toolkit
uv tool install .
```

## Quick example

```bash
dftk stat data.tsv -c height weight -g sex
dftk stat data.tsv -c value -g group -o \
  | dftk pivot -i group -v value_mean -o \
  | dftk print
```

## Subcommands

### Data transformation

| Command   | Description                                                                          |
|-----------|--------------------------------------------------------------------------------------|
| `eval`    | Add or modify columns: eval expressions, string/path functions, statistical ops      |
| `query`   | Filter rows with pandas query expressions or SQL (`--sql` via DuckDB)                |
| `merge`   | Join two tables on key columns (inner/left/right/outer)                              |
| `concat`  | Concatenate two or more tables row-wise                                              |
| `melt`    | Reshape wide-to-long (`pd.melt`)                                                     |
| `pivot`   | Reshape long-to-wide with per-cell aggregation                                       |
| `func`    | Column transforms: cumsum, group mean/sum/min/max/count/median/std, rank, qcut:N     |
| `scale`   | Normalise: z-score, min-shift, sum/max/mean scaling, Blom rank, regression residuals |
| `interp`  | Interpolate values from a reference curve into a table (1-D lookup)                  |
| `binx`    | Assign bin indices to a column based on explicit or generated edges                  |

### Statistics

| Command    | Description                                                                                         |
|------------|-----------------------------------------------------------------------------------------------------|
| `stat`     | Descriptive statistics (count/mean/std/CI/SEM/skew/kurtosis) with grouping and bootstrap            |
| `wstat`    | Weighted descriptive statistics (wmean, wstd, weighted quantile CI)                                 |
| `fit`      | OLS/robust/weighted regression via R-style formulas; tidy table, `--summary`, `--anova`             |
| `test`     | P-values between column pairs: t-test, Mann-Whitney, Wilcoxon, KS, correlations, bootstrap; groups  |
| `corr`     | Pairwise column correlations (Pearson/Spearman/Kendall) with optional BCa bootstrap CI              |
| `describe` | Quick column-level summary (dtype, n, n_unique, n_null, sample values)                              |
| `randvar`  | Sample from a distribution and append as a new column (norm, alpha, beta, …)                        |

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

| Command    | Description                                                              |
|------------|--------------------------------------------------------------------------|
| `dataset`  | Load a curated example dataset from seaborn, statsmodels, or pydataset   |
| `sample`   | Random row sampling (with or without replacement, by count or fraction)   |
| `split`    | Split a dataframe into one file per group                                 |
| `annotate` | Read and write provenance metadata (genome, source, …) in parquet files  |
| `print`    | Read any dftk input (TSV, stdin, `...` parquet pipe) and write TSV     |
| `clean`    | Remove leftover temp parquet pipe files from interrupted pipelines       |
| `help`     | List all subcommands or show full help for one: `dftk help stat`       |

## Common patterns

### Chaining commands

```bash
# z-score within group, then fit a model
dftk scale data.tsv -c expr -g condition -o \
  | dftk fit - -f "expr_z ~ time + batch" -g condition
```

### Group summary then plot

```bash
dftk stat results.tsv -c value -g group -o \
  | dftk line - -x group -y value_mean --yerr value_sem -f fig.png
```

### Wide-to-long then plot overlaid histograms

```bash
dftk melt data.tsv -i sample -d gene -v expression \
  | dftk hist - -x expression -g gene -k -f dist.png
```

### Interpolation (standard curve lookup)

```bash
dftk interp samples.tsv --ref stdcurve.tsv \
    -x fluorescence -v concentration -d conc_ng_ul
```

### Bootstrap confidence intervals

```bash
dftk stat data.tsv -c value -g group --bootstrap 1000 --randomseed 42 -o \
  | dftk pivot -i group -v value_mean -f mean -o \
  | dftk stat - -c group_A group_B
```

## Input/output

All commands accept:

- A TSV filename as a positional argument
- `-` to read TSV from stdin
- `...` to receive a parquet path from stdin (written by a previous `-o` command)
- A `.parquet` filename to read a named parquet file directly

Standard output options (available on all tabular commands):

- `-o` / `--output` — write a temp parquet for the next piped command instead of TSV to stdout
- `--select col1 col2 …` — keep only these columns
- `--drop col1 col2 …` — remove these columns
- `--round N` — round numeric output
- `--postquery EXPR` — filter output rows after processing
- `--meta KEY=VALUE` — embed provenance metadata in parquet output (repeatable)

### Provenance annotations

Metadata embedded with `--meta` is stored in the parquet file schema and propagates
automatically through the pipe: every subsequent `-o` write re-embeds it alongside
any new `--meta` values.

```bash
# Tag a file at creation
dftk eval raw.tsv -f "z = x + y" -o results.parquet \
    --meta genome=hg38 --meta source=gwas_2024

# Inspect annotations
dftk annotate results.parquet
# genome   hg38
# source   gwas_2024

# Add or update an annotation in-place
dftk annotate results.parquet --set step=qc_filtered

# Annotations survive piping
dftk eval results.parquet -f "z_scaled = z / 2" -o | dftk scale ... -c z -o scaled.parquet
dftk annotate scaled.parquet   # genome and source still present
```

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
- `statsmodels` — regression (`fit`, `scale --resid`, `wstat`)
- `duckdb` — SQL queries (`query --sql`)
- `matplotlib`, `seaborn` — plots
- `pyarrow` — parquet I/O backend

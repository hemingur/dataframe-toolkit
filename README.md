# dftk — DataFrame analysis and manipulation toolkit

`dftk` is a command-line toolkit for exploratory data analysis on TSV files.
Each subcommand reads tabular data, performs one operation, and writes TSV to
stdout, making it easy to chain commands in pipelines.

`dftk` was developed during computational genomics research as a fast,
composable alternative to writing one-off pandas scripts. It is designed
for analysts who live in the terminal and want a consistent, pipeable toolkit
for the full data analysis pipeline — from initial exploration through
statistical modelling and publication-quality figures.

**No data file handy?** Every example below uses `dftk dataset` to pull in a
small, real dataset, so you can copy-paste any of them and run them right
now. Run `dftk dataset --list` to browse everything available (dozens of
datasets from seaborn, statsmodels, and pydataset). `dftk dataset NAME`
searches all three sources in that order and returns the first match, so
`--source` is optional — pass it only to disambiguate a name that exists in
more than one source, or to restrict the search.

## Installation

```bash
uv tool install dataframe-toolkit
dftk --help
```

Or with `pip`:

```bash
pip install dataframe-toolkit
```

To install the latest `main` branch directly from GitHub:

```bash
uv tool install git+https://github.com/hemingur/dataframe-toolkit.git
```

To install from a local clone:

```bash
git clone https://github.com/hemingur/dataframe-toolkit.git
cd dataframe-toolkit
uv tool install .
```

### Skill for coding agents

`skills/dftk/` is an agent skill: a short guide that coding agents such as
Codex and Claude Code load when a task involves dftk. It covers the piping
conventions, which command to use for what, and patterns for Snakemake and
Make. Copy it into your agent's skills directory, for example:

```bash
cp -r skills/dftk ~/.codex/skills/     # Codex
cp -r skills/dftk ~/.claude/skills/    # Claude Code
```

Copy it again after upgrading dftk so it matches the installed version.

## Quick example

```bash
dftk dataset tips | dftk stat - -c tip total_bill -g day
```

Chain commands with `-o` (write to a temp parquet, print its path) and `...`
(read a parquet path from stdin) instead of piping TSV text directly — faster
for large data, and the recommended pattern once a pipeline has more than one
step:

```bash
dftk dataset tips -o \
  | dftk pivot ... -i day -v tip -f mean -o \
  | dftk print ...
```

## Subcommands

### Data transformation

| Command      | Description                                                                          |
|--------------|---------------------------------------------------------------------------------------|
| `eval`       | Add or modify columns: eval expressions, string/path functions, statistical ops      |
| `query`      | Filter rows with pandas query expressions or SQL (`--sql` via DuckDB)                |
| `merge`      | Join two tables on key columns (inner/left/right/outer)                              |
| `concat`     | Concatenate two or more tables row-wise                                              |
| `melt`       | Reshape wide-to-long (`pd.melt`)                                                     |
| `explode`    | Split separated values in a cell (`a,b`) into one row per value                      |
| `implode`    | Join a column's values within groups into one cell, the inverse of `explode`         |
| `pivot`      | Reshape long-to-wide with per-cell aggregation                                       |
| `func`       | Column transforms: cumsum, group mean/sum/min/max/count/median/std, rank, qcut:N     |
| `scale`      | Normalise: z-score, min-shift, sum/max/mean scaling, Blom rank, regression residuals |
| `interp`     | Interpolate values from a reference curve into a table (1-D lookup)                  |
| `binx`       | Assign bin indices to a column based on explicit or generated edges                  |
| `transpose`  | Flip rows and columns; original column names land in a new `--keycol` column         |
| `segid`      | Assign a segment ID that increments on each value change in a column                 |
| `cleannames` | Rename columns to lowercase identifiers (`Sample ID` → `sample_id`), e.g. for SQL    |

### Statistics

| Command    | Description                                                                                         |
|------------|-----------------------------------------------------------------------------------------------------|
| `stat`     | Descriptive statistics (count/mean/std/CI/SEM/skew/kurtosis) with grouping and bootstrap            |
| `wstat`    | Weighted descriptive statistics (wmean, wstd, weighted quantile CI)                                 |
| `fit`      | OLS/robust/weighted regression via R-style formulas; tidy table, `--summary`, `--anova`             |
| `test`     | P-values between column pairs: t-test, Mann-Whitney, Wilcoxon, KS, correlations, bootstrap; groups  |
| `corr`     | Pairwise column correlations (Pearson/Spearman/Kendall) with optional BCa bootstrap CI              |
| `describe` | Quick column-level summary (dtype, n, n_unique, n_null, sample values)                              |
| `info`     | Per-column dtype, null counts, and memory usage; `--summary` for dataset-level totals               |
| `randvar`  | Sample from a distribution and append as a new column (norm, alpha, beta, …)                        |

### Plots

All plot commands write a PNG/PDF when `-f FILE` is given, or display
interactively otherwise. They support `--groupcol` for colour grouping,
`--subgraphcol` for subplot grids, and figure/font presets for
publication-quality output.

| Command | Description                                                                                                                                                                         |
|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `scat`  | Scatter plot; OLS/robust or LOWESS fit (`--fit lowess`, one per group with `-g`), hexagonal binning for large tables (`--hexbin`), bubble size (`--sizecol`), colour (`--colorcol`) |
| `line`  | Line plot; optional error bars (`--yerr`) or CI bands (`--yci lo,hi`) and fit overlay                                                                                               |
| `hist`  | Histogram or KDE (`--kde`); normalisation, cumulative mode, mean±σ annotation (`--stats`)                                                                                           |
| `chi`   | Corner-resolved chi-plot: whether tail dependence sits in the lower or upper corner                                                                                                 |

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
# z-score sepal_length within species, then fit against petal_length
dftk dataset iris -o \
  | dftk scale ... -c sepal_length -g species -o \
  | dftk fit ... -f "sepal_length_scaled ~ petal_length" -g species
```

### Clean up spreadsheet headers

Headers such as `Country Name` or `1960` need quoting in SQL. `cleannames` turns them into lowercase identifiers (`country_name`, `x1960`) and keeps the original names in the parquet metadata (`dftk annotate FILE`):

```bash
dftk dataset fertility -o \
  | dftk cleannames ... -o \
  | dftk query ... --sql "SELECT country_name, x1960, x2000 FROM data LIMIT 3"
```

For your own export, `dftk cleannames data.tsv --backend duckdb -o data.parquet` also keeps integer columns with empty cells as integers.

### Large tables: hexbin instead of points

With tens of thousands of points a scatter plot is a solid blob. `--hexbin` counts points per hexagon instead (`--hexlog` for a log colour scale), and `--fit lowess` adds a smooth curve:

```bash
dftk dataset diamonds -o \
  | dftk scat ... -x carat -y price --hexbin --hexlog --logx --logy -f diamonds.png
```

`--colorcol COL` colours each hexagon by the mean of `COL` instead of the count; with `--subgraphcol` all panels share one colour scale.

### Group summary then plot

```bash
dftk dataset tips -o \
  | dftk stat ... -c tip -g day -o \
  | dftk line ... -x day -y mean --yerr sem -f fig.png
```

### Wide-to-long then plot overlaid histograms

```bash
dftk dataset iris \
  | dftk melt - -i species -d measurement -v value \
  | dftk hist - -x value -g measurement -k -f dist.png
```

### Interpolation (standard curve lookup)

```bash
printf "conc\tfluor\n0\t5\n10\t52\n20\t98\n30\t151\n" > stdcurve.tsv
printf "sample\tfluor\ns1\t60\ns2\t110\n" \
  | dftk interp - --ref stdcurve.tsv -x fluor --refx fluor -v conc -d conc_ng_ul
```

### Filling missing values

`eval -f` takes pandas column methods (`COL.method(...)`); `fillna`,
`ffill`, `bfill` and `interpolate` fill empty cells and keep the others.

| Formula | Fills with |
|---|---|
| `x = x.fillna(y)` | the value in column `y` |
| `x = x.fillna(0)`, `sex = sex.fillna('unknown')` | a fixed value |
| `x = x.fillna(y).fillna(0)` | `y` first, then 0 |
| `x = x.fillna(x.groupby(g).transform('mean'))` | the mean of the row's group (or `'median'`, ...) |
| `x = x.ffill()` / `x = x.bfill()` | the value above / below (`ffill(limit=1)`: at most one row) |
| `x = x.groupby(g).ffill()` | the value above, within each group of `g` |
| `x = x.interpolate()` | a straight line between the values above and below, by row position |

Write to a new column (`x_filled = x.fillna(y)`) to keep the original, and
mark the gaps first (`missing = x.isna()`) to tell filled values apart.
Only empty cells are filled: a missing row (a year with no row at all) is
not added.

**Defaults from another table.** A column that is in both files gets the
suffix `_r` from `merge`; fill from it, then drop it. Here the defaults are
each species' median body mass:

```bash
dftk dataset penguins -o | dftk pivot ... -i species -v body_mass_g \
  -f median --move median_body_mass_g:body_mass_g -o defaults.parquet
dftk dataset penguins -o \
  | dftk merge ... -r defaults.parquet -k species -t left -o \
  | dftk eval ... -f "body_mass_g = body_mass_g.fillna(body_mass_g_r)" \
      --drop body_mass_g_r
```

**From neighbouring rows** (`co2` is a weekly series with gaps):

```bash
dftk dataset co2 -o | dftk eval ... -f "ffill = co2.ffill()" \
  -f "bfill = co2.bfill()" -f "linear = co2.interpolate()"
```

`ffill` leaves the first rows empty and `bfill` the last ones;
`x.ffill().fillna(x.bfill())` fills both. `interpolate()` counts one step
per row, so the rows must be evenly spaced; for an x-based lookup from a
reference curve, see `dftk interp`.

**Row order and groups.** These methods use the rows in the order they are
in. Sort first, in a step of its own: `--sortasc` applies to a command's
output, so in the same command it would come after the fill. When a table
holds several series (sites, countries, samples), fill within each group,
or values are taken from the neighbouring group. Rows of a group don't need
to be next to each other, but within each group they must be in order:

```bash
dftk dataset healthexp -o \
  | dftk eval ... -f "le = Life_Expectancy.where(Year % 10 != 5)" \
      --sortasc Country Year -o \
  | dftk eval ... -f "le_filled = le.groupby(Country).ffill()"
```

`interpolate()` has no grouped form. Once each group's rows are together,
gaps inside a group are right, but a gap at a group's start or end is
filled from the neighbouring group. To leave those empty, keep only values
with something before and after them in the same group:

```bash
dftk dataset healthexp -o \
  | dftk eval ... -f "le = Life_Expectancy.where(Year % 10 != 5)" \
      --sortasc Country Year -o \
  | dftk eval ... -f "before = le.groupby(Country).ffill().notna()" \
      -f "after = le.groupby(Country).bfill().notna()" \
      -f "le_linear = le.interpolate().where(before & after)" \
      --drop before after
```

### Differences between rows

To compare a value with one in the previous or next row, for example the
gap between segments given by `Chr`, `Begin`, `End`, use SQL's `LAG()` /
`LEAD()` in `query --sql`:

```bash
dftk query segments.tsv --sql \
  'SELECT *, "Begin" - LAG("End") OVER (PARTITION BY Chr ORDER BY "Begin")
     AS gap
   FROM data ORDER BY Chr, "Begin"'
```

```text
Chr   Begin  End  Id  gap
chr1  100    200  a
chr1  250    400  b   50
chr1  390    500  c   -10
chr2  50     80   d
chr2  100    300  e   20
```

A negative gap is an overlap, and the first segment on each chromosome
gets an empty cell. `lead("Begin") OVER (...) - "End"` gives the gap to
the next segment instead.

- `Begin` and `End` are SQL keywords: quote such column names with double
  quotes and the whole statement with single quotes.
- No sorting is needed first; `ORDER BY` inside `OVER` sets the order.
- Integers stay integers.

`eval -f` can do the same with pandas `shift`, but it goes by row order, so
sort by `Chr` and `Begin` first (results are floats):

```bash
dftk eval segments.tsv -f 'gap = Begin - End.groupby(Chr).shift()'
```

Without `groupby(Chr)` the first segment of a chromosome is compared with
the last one of the previous chromosome.

### Bootstrap confidence intervals

Bootstrap mode repeats the full stat computation N times on resampled data
and emits one row per resample (tagged with `samplenum`) — pipe into `pivot`
with percentile aggfuncs (`cilo`/`cihi`) to collapse that into an empirical
confidence interval:

```bash
dftk dataset tips -o \
  | dftk stat ... -c tip -g day --bootstrap 1000 --randomseed 42 -o \
  | dftk pivot ... -i day -v mean -f mean cilo cihi -o \
  | dftk print ...
```

## Input/output

All commands accept:

- A TSV filename as a positional argument
- `-` to read TSV from stdin
- `...` to receive a parquet path from stdin (written by a previous `-o` command)
- A `.parquet` filename to read a named parquet file directly

> **`-` vs `...`** — these are not interchangeable. Use `-` when the previous
> command in the pipe wrote plain TSV to stdout (the default). Use `...` when
> the previous command used bare `-o` (pipe mode), which writes a temp
> parquet and prints *its path* to stdout — `-` would try to parse that path
> as TSV and fail.

### Comment lines and preambles

dftk has no comment or skip-rows option: `#` is ordinary data, so headers
like `Count #` and values like `A #1` are read as they are. Drop comment
lines or a preamble in the shell instead:

```bash
grep -v '^#' data.tsv | dftk print           # drop every line starting with #
tail -n +3 data.tsv | dftk print             # skip a 2-line preamble
dftk print <(grep -v '^#' data.tsv)          # the same, as a filename
```

With `--backend duckdb`, a whole-line `#` comment below the header is an
error, and `#` lines above the header are skipped.

Standard output options (available on all tabular commands):

- `-o` / `--output` — controls where output goes; takes an *optional* value:
  - `-o` alone — write a temp parquet to pipe into the next command, printing its path to stdout (auto-deleted once read)
  - `-o FILE.parquet` — write a named, reusable parquet, printing its path to stdout
  - `-o FILE` (no `.parquet` extension) — write TSV to `FILE`, with no stdout output
  - omit `-o` entirely — write TSV to stdout (the default)
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
dftk dataset iris -o iris.parquet --meta source=seaborn --meta species_col=species

# Inspect annotations
dftk annotate iris.parquet
# source        seaborn
# species_col   species

# Add or update an annotation in-place
dftk annotate iris.parquet --set step=raw

# Annotations survive piping
dftk eval iris.parquet -f "petal_area = petal_length * petal_width" -o iris2.parquet
dftk annotate iris2.parquet   # source, species_col, and step all still present
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

---
name: dftk
description: Use the dftk command-line toolkit to inspect, filter, join, reshape, summarise, test, fit and plot tabular data (TSV or parquet) from the shell, in pipelines, Makefiles or Snakemake rules. Use when a task involves dftk, or when tabular analysis should be done with shell commands rather than a Python script.
---

# dftk

`dftk` is a single CLI with one subcommand per operation on a table:

```text
dftk <command> [DATAFILE] [options]
```

Every data command reads one table (TSV or `.parquet`) and writes one table.
Install: `uv tool install dataframe-toolkit` (or `pip install dataframe-toolkit`).

This file covers what `--help` does not make obvious. For any command's full
options and worked examples, run `dftk <command> --help`. The help is
authoritative and up to date with the installed version; read it before
using a command for the first time.

## Rules that prevent silent mistakes

1. **Put DATAFILE first**, straight after the command name:
   `dftk stat data.tsv -c x`. Many options take a variable number of values
   (`-c`, `-g`, `--select`, `--drop` …), so a trailing filename is swallowed
   as one more column name, and dftk then reads stdin instead. `-o` also takes
   an optional value, so `dftk stat -c x -o data.tsv` *writes* to `data.tsv`.
2. **Piping between dftk commands uses `-o` and `...`**: end a command with a
   bare `-o` and give the next one the literal `...` as its DATAFILE.
   ```bash
   dftk query data.tsv -q "score > 0" -o | dftk stat ... -c score -g group
   ```
   A bare `-o` writes a temporary parquet file and prints only its *path*;
   `...` reads that path from stdin, loads the file and deletes it. Do not use
   `-` (or omit DATAFILE) after a bare `-o`: those read stdin as TSV data and
   would silently parse the printed path as a one-row table.
3. **Plain TSV piping** also works, without `-o` upstream and with no
   DATAFILE downstream: `dftk dataset tips | dftk print`. Prefer `-o | ...`
   between dftk commands: it keeps column types exact and is faster.
4. **Missing values are written as empty TSV fields**, not `NaN`/`NA`.
   When parsing dftk output in other tools, treat an empty field as missing.
5. **Plots need `-f FILE`** (`.png`, `.pdf`, `.svg`). Without it, dftk
   tries to open a window: with no display it prints a warning, writes
   nothing and still exits 0; with X forwarding it blocks until the window
   is closed.
6. **Check exit statuses**: a failing dftk step exits non-zero with an
   `Input error: …` message on stderr. In a pipe, later steps then fail with
   `'...' expects a .parquet path on stdin, got ''`; the *first* error is the
   real one. Use `set -o pipefail` in shell scripts (Snakemake does this by
   default).
7. **Integer columns with empty cells become float** with the default
   reader (`10` → `10.0`; IDs such as `007` become `7`). Read TSV with
   `--backend duckdb` to keep them integer (`Int64`) and IDs as text, e.g.
   when converting to parquet. For a spreadsheet export, do it with
   `cleannames`, which also fixes headers such as `Sample ID` or `GC %`:
   `dftk cleannames data.tsv --backend duckdb -o data.parquet`. Parquet keeps
   the types, so later steps need no option.
   For a column that is already float, `--cast COL:Int64` turns it back.
   `--digits N` writes such columns in exponent form (`2.49e+08`) and warns.

## Output modes (`-o`, on every data command)

| Form | Writes | Prints to stdout |
|------|--------|------------------|
| no `-o` | TSV to stdout | the table |
| `-o` | temp parquet in `$DFTK_TMPDIR` | its path (for `...`) |
| `-o FILE.parquet` | named parquet, kept | its path |
| `-o FILE` (any other extension) | TSV file | nothing |

`DFTK_TMPDIR` defaults to `$TMPDIR/dftk` (usually `/tmp/dftk`). Temp files
left by an interrupted pipe are removed with `dftk clean`.

## Options shared by every data command

Input: `--prequery EXPR` (filter rows right after reading), `--noheader`
(columns become `V1, V2, …`), `--delimiter`, `--readasobject COL…` (keep
values like IDs as strings).

Output, applied after the command's own work: `--select COL… [+]` (choose
and order columns; globs allowed; a trailing `+` keeps the remaining
columns), `--drop COL…`, `--move OLD:NEW…` (rename), `--postquery EXPR`,
`--sortasc COL…` / `--sortdesc COL…` / `--sort COL:a COL:d`,
`--round COL:N`, `--digits N`, `--cast COL:TYPE`, `--deduplicate [COL…]`,
`--dropna`, `--na_rep VAL`, `--removeheader`, and
`--meta KEY=VALUE` (provenance, see below).

## Choosing a command

Look at a table first:

| Need | Command |
|------|---------|
| Column names, dtypes, null counts | `dftk info FILE` |
| Distributions and data-quality flags per column | `dftk describe FILE` |
| Show a parquet file as TSV | `dftk print FILE` |

Transform:

| Need | Command |
|------|---------|
| New column from a formula: `z = x / y`, `where(...)`, string, DNA and statistical functions | `eval -f`, `eval -c` (constant), `eval -s` (string) |
| Fill empty cells: from another column, a group mean, the row above (`ffill`, per group with `groupby`) or by interpolation; sort first, fills follow row order | `eval -f "x = x.fillna(y)"`, `"x = x.groupby(g).ffill()"` (see `eval -h`) |
| Filter rows (pandas expression) or run any DuckDB SQL (`FROM data`) | `query -q` / `query --sql` |
| Join two tables, including anti-joins | `merge DATAFILE -r OTHER -k KEY [-t left] [--only left]` |
| Stack tables row-wise | `concat A B C [--sourcecol src]` |
| Wide to long / long to wide plus aggregation | `melt` / `pivot -i ROWS -g COLS -v VALS -f FUNCS` |
| Split comma-separated values in a cell into one row each, repeating the other columns | `explode -c COL [--sep SEP]` |
| Swap rows and columns | `transpose` |
| Rename columns to lowercase identifiers (`Sample ID` → `sample_id`, `GC %` → `gc_pct`) so SQL, `eval` and formulas need no quoting | `cleannames [--mapping FILE]` |
| Per-group values broadcast to every row (group mean, rank, cumsum, quantile bin) | `func -c COL -t TRANSFORM -g GROUP` |
| Bin a numeric column | `binx -c COL -b 0:100:10` |
| Normalise, rank-transform or detrend columns | `scale` |
| Look up y at x from a reference curve | `interp DATAFILE --ref CURVE -x X -v Y` |
| Run-length segment IDs | `segid` |
| Random sample, optionally per group | `sample -n N` / `sample -f FRAC [-g G]` |
| One output file per group | `split -g G --template 'out/{G}.tsv'` |
| Add a column of random variates | `randvar` |

Statistics (all accept `-g` for per-group results):

| Need | Command |
|------|---------|
| count, mean, std, median, CI, sem, skew … | `stat -c COL…` (`--bootstrap N` available) |
| Weighted versions | `wstat` |
| Linear, robust and weighted regression (R-style formula) | `fit -f "y ~ x + z"` |
| p-values for column pairs (t, Mann-Whitney, KS, …) | `test -c A:B -d p --test NAME` |
| Correlations with CIs, bootstrap, tail dependence | `corr -c A:B` |

Plots (always with `-f FILE`): `scat` (scatter; `--fit` for an OLS line,
`--fit lowess` for a smooth curve, per group with `-g`; `--hexbin` for
tables too large for points), `line`, `hist`
(also KDE), `chi` (tail-dependence chi-plot). `-g COL` overlays groups,
`--subgraphcol COL` makes panels.

Other: `dataset NAME` loads a bundled example table, `annotate FILE.parquet`
reads and writes provenance metadata, `clean` removes leftover temp pipe
files.

## Examples

```bash
# Inspect
dftk info results.tsv

# Filter, derive a column, summarise per group
dftk query results.tsv -q "qual >= 30" -o \
  | dftk eval ... -f "ratio = alt_count / depth" -o \
  | dftk stat ... -c ratio -g sample

# SQL when a pandas expression gets awkward
dftk query results.parquet --sql \
  "SELECT sample, COUNT(*) AS n, AVG(depth) AS mean_depth
   FROM data GROUP BY sample ORDER BY n DESC"

# Join with an annotation table, keeping unmatched rows
dftk merge results.tsv -r annotation.tsv -k gene -t left -o merged.parquet

# Per-group regression, then keep only the slope rows
dftk fit merged.parquet -f "expression ~ dose" -g gene \
  --postquery "marker == 'dose'"

# Scatter plot coloured by group
dftk scat merged.parquet -x dose -y expression -g batch -f dose.png
```

Try a pipeline on real data before writing it against your own files: the
bundled datasets (`dftk dataset --list` lists them; `tips`, `penguins`,
`iris`, `diamonds` are good defaults) work anywhere, e.g.
`dftk dataset penguins -o | dftk stat ... -c body_mass_g -g species sex`.

## Provenance metadata

Parquet output can carry key/value annotations that survive later steps:

```bash
dftk query raw.parquet -q "pass_qc" -o clean.parquet --meta step=qc --meta build=hg38
dftk annotate clean.parquet            # list all annotations
dftk annotate clean.parquet --get build
```

Annotations read from an input are written again on every parquet output;
`--meta` adds keys or overrides them. TSV output has no metadata.

## Make and Snakemake

For rules in workflows (named outputs, wildcards, logs, provenance, when to
pipe inside a rule and when to split it), read
[references/snakemake.md](references/snakemake.md).

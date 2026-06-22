# stattools — CLAUDE.md

## Project overview

`stattools` is a rewrite/port of the legacy `df` package located at
`/home/gunnar/projects/python_projects/df/src/df/`.  Each original `df*.py`
script becomes a `dfstat <subcommand>` in stattools.  When porting, consult the
original script for algorithm details and edge cases, but redesign the CLI
interface where it was confusing (e.g. `interp_cmd` renamed left/right to
data/ref).

`stattools` is a CLI toolkit for DataFrame analysis and manipulation, exposed as a single entry-point:

```text
dfstat <subcommand> [options] [DATAFILE]
```

Built with Python 3.12+, pandas 2.x, numpy, scipy, statsmodels, duckdb, seaborn/matplotlib.
Package manager: **uv**. Run tests with `pytest` (no special flags needed).

---

## Project layout

```text
src/stattools/
  cli.py                    # Entry-point; discovers commands from commands/COMMANDS list
  commands/
    __init__.py             # COMMANDS registry — import class here + add instance to command_list
    base.py                 # BaseCommand ABC (name, help, add_arguments, execute)
    eval_cmd.py             # dfstat eval   — formula / constant / string functions
    stat_cmd.py             # dfstat stat   — descriptive statistics
    pivot_cmd.py            # dfstat pivot  — pivot / reshape
    merge_cmd.py            # dfstat merge  — join two files
    query_cmd.py            # dfstat query  — SQL-style filter (duckdb)
    fit_cmd.py              # dfstat fit    — curve fitting / regression
    scale_cmd.py            # dfstat scale  — normalisation / scaling
    func_cmd.py             # dfstat func   — column transforms (cumsum, rank, qcut, groupby aggs)
    interp_cmd.py           # dfstat interp — interpolate values from a reference curve
    dataset_cmd.py          # dfstat dataset  — load bundled/seaborn/statsmodels example datasets
    annotate_cmd.py         # dfstat annotate — read/write parquet provenance metadata
    melt_cmd.py, scat_cmd.py, line_cmd.py, hist_cmd.py, print_cmd.py, clean_cmd.py, ...
  common/
    io.py                   # io.read(args), io.printdf(df, args) — TSV in/out
    stats.py                # Statistical helpers (binom_test, fisher_test, etc.)
    seq.py                  # DNA/sequence utilities
tests/
  conftest.py               # make_args(**kwargs) helper used by all tests
  commands/test_eval.py     # unit tests for eval_cmd
  commands/test_func.py     # unit tests for func_cmd
  commands/test_interp.py   # unit tests for interp_cmd
  commands/test_dataset.py  # unit tests for dataset_cmd
  commands/test_stat.py     # ... and so on
  common/                   # tests for common utilities
```

---

## Adding a new subcommand

1. Create `src/stattools/commands/<name>_cmd.py` with a class inheriting `BaseCommand`.
2. Implement `name`, `help`, `add_arguments(parser)`, `execute(args)`.
3. Import the class in `commands/__init__.py` and append an instance to `command_list`.
4. Write tests in `tests/commands/test_<name>.py`.

Standard I/O pattern:

```python
def execute(self, args):
    df = io.read(args)       # reads DATAFILE positional arg (or stdin)
    # ... transform df ...
    io.printdf(df, args)     # writes TSV to stdout or -o file
```

---

## Key conventions

- **File naming**: all command modules are `<name>_cmd.py` (e.g. `stat_cmd.py`, not `stat.py`).
- **TSV I/O**: `io.printdf` writes NaN as empty string (trailing tab). Test helpers must convert `""` → `float("nan")` when parsing TSV output.
- **groupby keys**: `df.groupby(["col"])` with a list always returns tuple keys even for a single column. Use the same groupby call on both sides so keys match directly.
- **Tests use `make_args`**: imported from `tests/conftest.py`; creates a simple namespace for passing args to `_eval`, `_func`, etc. without going through the CLI parser.

---

## eval_cmd internals

Three formula modes:

- `-f EXPR`: pandas `df.eval()` first; on failure falls back to `_special_function()`
- `-c EXPR`: constant column assignment (`dest = value`, value coerced to int/float/str)
- `-s EXPR`: string / row functions

**Special functions** available in `-f` (fallback path):

- Row aggregation: `sum mean std min max median` (glob patterns supported)
- Row index: `idxmax idxmin` (returns column name of max/min per row)
- Conditional: `where(cond_col, true_val, false_val)` — np.where-style; each value resolved as column name, quoted string `"foo"`, or numeric literal
- Bitwise: `applymask overlap`
- Column glob: `colsum(col*)`
- NumPy: `sign`
- Path: `basename dirname exists getsize realpath`
- Stats: `binom_test fisher_test fisher_OR boschloo_test pval2se t2pval chi2_to_neglogp neglogp_to_chi2 pval_to_chi2 generalized_poisson_nll`

**Dispatch classes** (`_FrameFunc`, `_WhereFunc`): used when the function needs the full DataFrame rather than a single row/column. Detected by `isinstance` checks in `_eval`.

---

## func_cmd key points

- `-t/--transform`: `cumsum`, `sum/mean/min/max/count/median/std` (group broadcast), `rank`, `pct_rank`, `qcut:N`
- `-g/--groupcol` enables groupby; group aggregates are broadcast back to original index
- Default destcol: `{col}_{transform}` with special chars replaced by `_`

---

## annotate_cmd / parquet metadata

Parquet files store arbitrary key-value string pairs in the file-level schema metadata.
dfstat uses this for provenance tracking.

Key functions in `common/io.py`:

- `_write_parquet(df, path, meta=None)` — writes via pyarrow, merging `df.attrs["_parquet_meta"]` (carried from prior reads) with explicit `meta` dict
- `_read_parquet_meta(path)` — reads all non-`pandas` keys from schema metadata

Propagation: `io.read()` stores custom metadata in `df.attrs["_parquet_meta"]`; `_write_parquet` re-embeds it on every parquet write, so annotations survive multi-step pipelines.

The `--meta KEY=VALUE` flag (available on all commands via `parser_output`) embeds metadata at write time and takes precedence over carried attrs.

`annotate_cmd.py` is not a data-transform command — it operates directly on a parquet file path (no `io.read`/`io.printdf`). Default action (no flags) lists all annotations as sorted TSV.

---

## interp_cmd key points

- `DATAFILE` = data to enrich (query side); `--ref` = reference curve
- `-x/--xcol` = x column in data; `--refx` = x column in reference (defaults to same as `-x`)
- `-v/--val` and `-d/--destcol` support multiple values (zip with destcols)
- Uses `scipy.interp1d`; `--fill {nan,edge}` controls out-of-bounds behaviour
- Known limitation: flat reference segments cause round-trip interpolation ambiguity — requires manual handling by the caller

---

## Ported / pending subcommands

Original scripts live in `/home/gunnar/projects/python_projects/df/src/df/`.

| Original script  | dfstat subcommand | Status    |
|------------------|-------------------|-----------|
| dfstat.py        | stat              | ported    |
| dfeval.py        | eval              | ported    |
| dfpivot.py       | pivot             | ported    |
| dfmerg.py        | merge             | ported    |
| dfquery.py       | query             | ported    |
| dfsmfit.py       | fit               | ported    |
| dfscale.py       | scale             | ported    |
| dfscat.py        | scat              | ported    |
| dfline.py        | line              | ported    |
| dfhist.py        | hist              | ported    |
| dfmelt.py        | melt              | ported    |
| dffunc.py        | func              | ported    |
| dfipol.py        | interp            | ported    |
| —                | dataset           | new       |
| dfsample.py      | sample            | ported    |
| dfcat.py         | concat            | ported    |
| dfsplit.py       | split             | pending   |
| dfcorr.py        | corr              | pending   |
| dftest.py        | test              | ported    |
| dfwstat.py       | wstat             | pending   |
| dfbinx.py        | binx              | pending   |
| dfsegid.py       | segid             | pending   |
| dfrvs.py         | randvar           | ported    |
| dfinfo.py        | info              | pending   |
| dftpos.py        | transpose         | pending   |
| dfcolor.py       | color             | pending   |
| dffisher.py      | fisher            | pending*  |
| dfbspl.py        | —                 | pending   |
| dfrsfit.py       | —                 | pending   |

\* `fisher_test`, `fisher_OR`, and `boschloo_test` are already available as functions in `eval` — a dedicated port is likely unnecessary.

# dftk — CLAUDE.md

## Project overview

`dftk` (package `dataframe-toolkit`) is a rewrite/port of the legacy `df` package located at
`~/projects/python_projects/df/src/df/`.  Each original `df*.py`
script becomes a `dftk <subcommand>` in dftk.  When porting, consult the
original script for algorithm details and edge cases, but redesign the CLI
interface where it was confusing (e.g. `interp_cmd` renamed left/right to
data/ref).

`dftk` is a CLI toolkit for DataFrame analysis and manipulation, exposed as a single entry-point:

```text
dftk <subcommand> [options] [DATAFILE]
```

Built with Python 3.12+, pandas 2.x, numpy, scipy, statsmodels, duckdb, seaborn/matplotlib.
Package manager: **uv**. Run tests with `pytest` (no special flags needed).

---

## Project layout

```text
src/dftk/
  cli.py                    # Entry-point; discovers commands from commands/COMMANDS list
  commands/
    __init__.py             # COMMANDS registry — import class here + add instance to command_list
    base.py                 # BaseCommand ABC (name, help, add_arguments, execute)
    eval_cmd.py             # dftk eval   — formula / constant / string functions
    stat_cmd.py             # dftk stat   — descriptive statistics
    pivot_cmd.py            # dftk pivot  — pivot / reshape
    merge_cmd.py            # dftk merge  — join two files
    query_cmd.py            # dftk query  — SQL-style filter (duckdb)
    fit_cmd.py              # dftk fit    — curve fitting / regression
    scale_cmd.py            # dftk scale  — normalisation / scaling
    func_cmd.py             # dftk func   — column transforms (cumsum, rank, qcut, groupby aggs)
    interp_cmd.py           # dftk interp — interpolate values from a reference curve
    dataset_cmd.py          # dftk dataset  — load bundled/seaborn/statsmodels example datasets
    annotate_cmd.py         # dftk annotate — read/write parquet provenance metadata
    melt_cmd.py, scat_cmd.py, line_cmd.py, hist_cmd.py, print_cmd.py, clean_cmd.py, ...
    chi_cmd.py               # dftk chi    — corner-resolved chi-plot (tail-dependence diagnostic)
  common/
    io.py                   # io.read(args), io.printdf(df, args) — TSV in/out
    stats.py                # Statistical helpers (binom_test, fisher_test, etc.)
    seq.py                  # DNA/sequence utilities
    copula.py               # Nonparametric tail-dependence diagnostics (used by corr --tail, chi)
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

1. Create `src/dftk/commands/<name>_cmd.py` with a class inheriting `BaseCommand`.
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

## Git workflow

- **Branches**: use a feature branch for any change that touches multiple files or adds a new subcommand. Merge via PR. Direct commits to `main` are fine for single-file, low-risk changes (typo fix, one new test).
- **Commit size**: one logical change per commit. A new subcommand warrants at least two commits: implementation and tests.
- **Commit prefixes** (same style as genomics-tool):
  - `feat:` — new subcommand or significant new capability
  - `fix:` — bug fix
  - `test:` — adding or updating tests only
  - `chore:` — tooling, deps, CI, version bumps
  - `docs:` — CLAUDE.md, README, comments
  - `refactor:` — code restructuring with no behaviour change
- **No `Co-Authored-By` trailers** in commit messages.

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
dftk uses this for provenance tracking.

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

## copula.py / corr --tail / chi key points

Nonparametric tail-dependence diagnostics, domain-agnostic (equally
applicable to genetics, finance, climate, etc.) — not a genomics-specific
feature despite the original design chat leaning on genomics examples.

- `corr --tail --tail-q Q1 Q2 ...` adds `lambda_lower_qN`/`lambda_upper_qN`
  columns per pair. `chi` is the visual counterpart: a corner-resolved
  chi-plot, structurally a near-mirror of `scat_cmd.py`.
- `corner_chi()` deliberately deviates from the textbook Fisher & Switzer
  chi-plot: the textbook version signs its x-axis by "concordant vs
  discordant", which cannot distinguish lower-tail dependence (e.g. Clayton
  copula) from upper-tail dependence (e.g. Gumbel) — both land on the same
  side. This version restricts to concordant pairs and signs x by which
  corner each point is actually in, so the two corners are genuinely
  separated. Validated against synthetic Clayton-copula data with a known
  theoretical tail-dependence coefficient (see `tests/common/test_copula.py`).
- `bivariate_loo_cdf()` (the leave-one-out empirical CDF behind `corner_chi`)
  must be tie-safe: ties in real data (rounded/binned values, repeated
  measurements) are common, not an edge case, and a tie-unsafe O(n log n)
  implementation silently biases results rather than crashing — validate any
  change here against the naive O(n^2) reference in
  `tests/common/test_copula.py::TestBivariateLooCdf` before trusting it.
- `chi --groupcol` computes a *separate* chi statistic per group and
  overlays them as distinct series (matching how `scat`/`hist` overlay group
  series) rather than one summary row per group like `corr -g` — mixing
  groups into one leave-one-out CDF would be statistically meaningless,
  since the CDF must be computed within a single joint distribution.

### Possible extension: parametric copula fitting (BB1 / threshold-split)

`--tail` and `chi` are nonparametric diagnostics only — they characterize
tail-dependence *shape* but don't fit a parametric copula. A natural next
step, not yet implemented, is a fitting mode (e.g. `corr --fit-copula`)
that fits a named copula and reports its parameter(s).

Worked example that motivated this (`dftk corr` on the bundled `diamonds`
dataset, `-c carat:price --tail`): lower-tail dependence present
(`lambda_lower` plateaus ~0.28–0.33 across q), upper-tail dependence absent
(`lambda_upper` decays toward 0 as q shrinks). That asymmetric shape is
exactly Clayton's signature (`lambda_L = 2^(-1/theta) > 0`, `lambda_U = 0`)
— but a plain one-parameter Clayton fit is the wrong tool here:

- Clayton's single `theta` controls both tail strength *and* bulk
  association (Kendall's tau = `theta / (theta + 2)`) simultaneously.
- `carat:price` has very high bulk association (Pearson r=0.92, Kendall's
  tau almost certainly similarly high) but only moderate lower-tail
  dependence (~0.3). Calibrating `theta` to the bulk tau would predict far
  stronger tail clustering than observed; calibrating to the tail would
  understate the bulk correlation. One parameter can't satisfy both at once.

Two ways to actually handle this, if/when it's implemented:

1. **BB1** (Joe's two-parameter Archimedean family) — has independent
   parameters for tail strength and overall association, so it can match
   both facts at once instead of conflating them. The right family for
   "high bulk correlation + moderate, one-sided tail dependence" cases like
   this one.
2. **Threshold-split fit** — fit separate (possibly simpler) copulas or
   correlation estimates above/below a chosen split point on one variable,
   rather than forcing one global copula. Often the more honest choice when
   the BB1-vs-Clayton mismatch itself signals that dependence strength
   varies with the *level* of the variables — e.g. `carat:price`: small
   diamonds cluster near a practical price/size floor, while large-diamond
   pricing is driven more by cut/clarity/color than by carat, so dependence
   genuinely weakens at the high end. That's closer to heteroskedastic
   dependence than to a fixed-copula tail effect — a single global copula
   (BB1 included) would still average over it rather than represent it.

Not scheduled; documented here so the reasoning survives if picked up later.

---

## Ported / pending subcommands

Original scripts live in `~/projects/python_projects/df/src/df/`.

| Original script  | dftk subcommand   | Status    |
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
| dfsplit.py       | split             | ported    |
| dfcorr.py        | corr              | ported    |
| —                | chi               | new       |
| dftest.py        | test              | ported    |
| dfwstat.py       | wstat             | ported    |
| dfbinx.py        | binx              | ported    |
| dfsegid.py       | segid             | ported    |
| dfrvs.py         | randvar           | ported    |
| dfinfo.py        | info              | ported    |
| dftpos.py        | transpose         | ported    |
| dfcolor.py       | color             | pending   |
| dffisher.py      | fisher            | pending*  |
| dfbspl.py        | —                 | pending   |
| dfrsfit.py       | —                 | pending   |
| dfwtpwr.py       | wavelet           | pending   |

\* `fisher_test`, `fisher_OR`, and `boschloo_test` are already available as functions in `eval` — a dedicated port is likely unnecessary.

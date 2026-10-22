# Changelog

## [0.10.0] — 2026-10-22

### Added
- `dftk explode`: splits a column of separated values (`GO:1, GO:2`) into one row per value, repeating the other columns (`--select` keeps only some). `--sep` (default `,`) is a plain string, not a regex, and spaces around items are removed. Several `-c` columns are split in parallel, their n-th items on the same row; a row with different item counts is an error. An empty cell stays as one row unless `--dropempty`, and `--poscol NAME` numbers each item's position in its cell. Split columns whose values are all numbers become numeric, except columns named in `--readasobject`.

## [0.9.1] — 2026-10-20

### Fixed
- `merge`: integer columns turned into floats (`100.0`) when a left, right or outer join left rows unmatched, including with `--backend duckdb` and parquet input. Integer columns on the side that can get gaps are now made nullable (`Int64`) before the join, so they stay integers; the `-k` key columns are not changed. `--only left`/`right` also kept the chosen side's integer columns as floats, because the outer join it runs first left gaps in them.
- `--backend duckdb` read a pipe or process substitution (`dftk print <(grep -v '^#' data.tsv) --backend duckdb`) as a header with no rows, without an error: DuckDB reads its input twice, and the first read used up the pipe. Such input is now copied to a temporary file first, as stdin already was.
- `--backend duckdb` treated `#` as a comment character in a file with a whole-line `#` comment below the header. It cut every line at `#`, so a `Count #` header became `Count` (or the header was lost) and values like `A #1` were cut short, without an error. DuckDB no longer guesses a comment character: such a file is now an error, and `grep -v '^#'` drops the comment lines (see the README). `#` lines above the header are still skipped.

## [0.9.0] — 2026-10-17

### Added
- `scat --hexbin`: hexagonal binning for tables too large for a scatter plot (port of the legacy `dfhexb`). `--gridsize N` (default 100), `--hexlog` for a log colour scale, and `--colorcol COL` colours each hexagon by the mean of `COL` instead of the count. With `--logx`/`--logy` the hexagons are binned on the log scale (the legacy script binned first and then changed the axes, which distorted them); rows with values ≤ 0 are dropped with a message. With `--subgraphcol` all panels share one colour scale. Cannot be combined with `-g` or `--sizecol`.
- `scat --fit lowess`: a LOWESS curve (local regression, statsmodels) instead of a straight line, with `--frac F` (default 2/3) for how much of the data each local fit uses. It fits only every 1% of the x range and interpolates between, so a million points take seconds. A bare `--fit` is still an OLS line. This replaces the legacy `dfbspl`, whose interpolating spline passed through every point.

### Changed
- `scat --fit` with `-g` draws one line (or curve) per group in the group's colour, over that group's x range, and prints a regression summary per group. Before, `--fit` was silently ignored when `-g` was given.

## [0.8.0] — 2026-10-17

### Added
- `dftk cleannames`: renames columns to lowercase ASCII identifiers, so spreadsheet headers can be used unquoted in `query --sql`, `eval -f` and `fit` formulas: `Sample ID` → `sample_id`, ` Age (years) ` → `age_years`, `GC %` → `gc_pct`, `Count #` → `count_n`, `1960` → `x1960`, `Hæð (µg)` → `haed_ug`. camelCase is not split (`HbA1c` → `hba1c`), clashes get `_2`, `_3`, and names that are SQL reserved words (`group`, `order`) are kept with a warning. The original names are stored in the parquet metadata (`original_names`, see `dftk annotate`) and, with `--mapping FILE`, in a TSV file. Converting an export in one step: `dftk cleannames data.tsv --backend duckdb -o data.parquet`.

### Fixed
- Empty and duplicate headers in a TSV file were kept as they were by the default reader, so e.g. `dftk info` crashed on an export with two `Count #` columns. They are now named as pandas' standard reader names them (`Unnamed: 7`, `Count #.1`).

## [0.7.3] — 2026-10-15

Integer columns with empty cells, which pandas reads as float (`10.0`). The way to keep them integer is to read the TSV with `--backend duckdb` (they become `Int64`); parquet files keep the type, so later steps need no option:

```bash
dftk eval data.tsv --backend duckdb -o data.parquet
```

### Fixed
- `--backend duckdb` was silently ignored when reading stdin; `cat data.tsv | dftk eval - --backend duckdb …` now reads with DuckDB like a named file.
- `fit` and `scale --resid` crashed with `Unexpected error (TypeError): boolean value of NA is ambiguous` on `Int64` columns with empty cells (e.g. read with `--backend duckdb`). They are now passed to statsmodels as float, and incomplete rows are dropped as before.
- `--cast COL:Int64` turns a float column with empty cells back into integers (`248956422`, not `248956422.0`). `--cast` with `object`, `category` or `datetime64` also failed (or warned) before and now works. Type prefixes are case-sensitive: `int` is `int64`, `Int` is `Int64`.
- TSV output with a single column crashed when a cell was empty (`single empty field record must be quoted`, from Python's csv module). The missing value is now written as an empty line, and the default reader reads an empty line in a one-column file as a missing value instead of skipping it, so such files survive a round trip (DuckDB already read them this way). Files with more columns are read as before.
- `query --sql` returned `SUM` of an integer column as float (`19.0`), because DuckDB sums into a 128-bit integer that pandas has no type for. It is now an integer, up to 2^53 (beyond that the float may already be rounded, so it stays float).

### Changed
- `--digits N` warns when it writes a whole number in exponent form (`248956422` → `2.49e+08`), which happens to integer columns that were read as float. The warning names the columns and suggests `--cast COL:Int64` or `--round N`; the output itself is unchanged.
- `--dropna` help now says it drops rows from the output, after the command has run (use `--prequery` to filter the input).
- The agent skill (`skills/dftk`) has a new rule about integer columns with empty cells.

## [0.7.2] — 2026-10-14

### Added
- Agent skill in `skills/dftk/` for coding agents such as Codex and Claude Code: the piping conventions (`-o | ...`, DATAFILE first, `-f` for plots), a guide to which command to use for what, and Snakemake/Make patterns with a tested example workflow. Copy it into your agent's skills directory (see the README). A test checks every command and option in it against the real CLI, so it cannot drift silently.

### Fixed
- `fit` (and `scale --resid`) formulas can now use numpy functions, e.g. `-f "np.log(y) ~ x"`; before, they failed with `NameError: name 'np' is not defined`. A formula naming an unknown column or function is now reported as `Input error: cannot evaluate formula …` (exit 1) instead of an "unexpected error" whose traceback itself crashed on Python 3.14.
- `concat --fill` filled every empty cell, including cells that were already empty within an input file. It now fills only the columns a file lacks, as its help text says, and integer columns stay integer (`0`, not `0.0`).
- `stat` without `-g` printed `count` as a float (`244.0`); it is now an integer, as in grouped output.

## [0.7.1] — 2026-10-13

### Changed
- Faster startup: dftk now imports only the subcommand being run instead of all of them, so commands no longer pay for other commands' scipy / statsmodels / matplotlib imports. Typical startup drops from ~2.2 s to ~0.65 s for `print`, `query`, `pivot`, `eval` and the other data commands (measured on a loaded 2-core machine; about a 3× speed-up), which adds up quickly in multi-step pipelines. Commands that need those libraries gain less (`corr` ~25%, `scat` ~30%, `fit` ~10%); `dftk --help` is unchanged. `eval` imports scipy only when a formula uses one of its special or statistical functions.

## [0.7.0] — 2026-09-28

### Fixed
- `dftk ... | head` exited with status 120 (sometimes with an "Exception ignored while flushing sys.stdout: BrokenPipeError" message) once output exceeded the pipe buffer. dftk now exits quietly with status 141, like `cat`/`grep`. The old per-writer workaround also swallowed genuine write errors on named `-o` files; those now surface.
- `--backend duckdb` failed on every named TSV/CSV file (`query(): incompatible function arguments`). It now reads via `duckdb.read_csv`, and also honours `--noheader`.
- `pivot -z` did not fill empty cells when spreading columns with `-g` (and was a no-op with a single `-i`), so counts came out as floats with blanks. `-z` now passes `fill_value=0, dropna=False` to `pandas.pivot_table`, expanding both rows and columns to every combination and keeping integer columns integer. As a side effect of `dropna=False`, rows with NaN in a key column now form their own group instead of being dropped.

### Removed
- `--nrows` (all commands, including `concat`). It never worked: the default pandas/pyarrow reader rejects `nrows`, and the parquet and duckdb read paths silently ignored it. Use `head -n N+1 file.tsv | dftk ...` for TSV, or `dftk query file.parquet --sql "SELECT * FROM data LIMIT N"` for parquet (read lazily by DuckDB, so the whole file is not loaded).

## [0.6.0] — 2026-08-29

### Added
- `dftk corr --tail --tail-q Q1 Q2 ...`: adds `lambda_lower_qN`/`lambda_upper_qN` columns per requested threshold — nonparametric tail-dependence coefficient estimates, describing whether the *extremes* of two columns move together, separately from overall correlation. Domain-agnostic: applies equally to genetic markers, asset returns, or climate variables.
- `dftk chi`: new plot command, a corner-resolved chi-plot — the visual complement to `--tail`, showing whether tail dependence (if any) sits in the lower corner, upper corner, both, or neither, without picking a specific `q` threshold.
- `src/dftk/common/copula.py`: shared nonparametric copula diagnostics (`pseudo_observations`, `bivariate_loo_cdf`, `empirical_tail_dependence`, `corner_chi`) backing both of the above. `bivariate_loo_cdf` uses a tie-safe Fenwick-tree leave-one-out empirical CDF, O(n log n), validated against a naive O(n^2) reference across tied and untied data up to realistic table sizes.

## [0.5.2] — 2026-07-22

### Fixed
- `-xm`/`--xlim` on `scat`/`line`/`hist` no longer leaves the y-axis fixed to the full-dataset autoscale. Matplotlib computes y-limits from all plotted data when the artists are created, before `-xm` narrows the x range; `apply_limits()` now recomputes y-limits from only the data visible within the new x range (unless `-ym` is given explicitly).
- `--xmargin`/`--ymargin` were silently discarded when combined with an explicit `-xm`/`-ym`, because margins were applied before limits. Margins now apply after limits.

## [0.5.1] — 2026-07-22

### Fixed
- `stat -g ... --bootstrap ...` crashed with a `KeyError` on the group column — pandas 3.0 changed `groupby(cols, group_keys=False).apply(fn)` to drop the grouping column from the result unless explicitly reselected. `pivot`'s own `--bootstrap` path already worked around this; applied the same fix to `stat`.

### Changed
- Rewrote README examples to be self-contained and verified runnable: every example now uses `dftk dataset` instead of an unspecified `data.tsv`, and every command block was actually executed end-to-end before being committed.
- Fixed a systemic `-` vs `...` mixup across nearly all piped `-o` examples in the README (and in `stat_cmd.py`'s own module docstring) — `-` reads literal TSV from stdin, `...` reads a parquet path from stdin written by a previous `-o`; using the wrong one fails with a pandas/pyarrow error that doesn't point at the actual mistake.
- Documented that `dftk dataset NAME` searches all sources (seaborn, statsmodels, pydataset) in order and returns the first match — `--source` is optional.
- Added `info`/`transpose`/`segid` to the README's subcommand tables (implemented in 0.5.0 but never documented there).

## [0.5.0] — 2026-07-22

### Added
- New subcommand: `info` — per-column dtype/null/memory overview (TSV), with `--summary` for dataset-level totals. Redesigned from the legacy `dfinfo.py`, which just printed pandas' `df.info()` as text; complements `describe`'s statistical profiling.
- New subcommand: `transpose` — flips rows and columns; original column names land in a new `--keycol` column (default: `column`).
- New subcommand: `segid` — assigns a segment ID that increments on each value change in a column, for grouping contiguous runs. `--ignore VALUE` excludes matching rows (segid 0) without breaking a run spanning across them.

### Fixed
- `segid`'s run-change detection uses `.ne(.shift())` instead of the legacy `.diff().ne(0)` — `diff()` computes subtraction, which the current pandas/pyarrow string dtype doesn't support, so the original approach would crash on categorical/string columns (the most common real use case).

### Changed
- `segid`'s `--ignore` now defaults to `None` (segments every row) instead of the legacy script's magic sentinel string (`'1415926535'`, digits of pi) standing in for "nothing ignored".

## [0.4.1] — 2026-07-22

### Added
- First public release on PyPI (`pip install dataframe-toolkit` / `uv tool install dataframe-toolkit`)
- GitHub Actions publish workflow (trusted publishing / OIDC, no stored tokens): builds on GitHub Release, publishes to TestPyPI and PyPI

## [0.4.0] — 2026-07-22

### Changed
- Project renamed: console script `dfstat` → `dftk`; PyPI distribution `stattools` → `dataframe-toolkit`; import package `stattools` → `dftk`
- `DFSTAT_TMPDIR` env var renamed to `DFTK_TMPDIR`
- Claude subagents renamed: `stattools-code-reviewer`/`stattools-test-writer` → `dftk-code-reviewer`/`dftk-test-writer`
- Reason: `stattools` clashed with an unrelated existing PyPI package, and overstated the tool's scope — most subcommands (`merge`, `func`, `eval`, `pivot`, …) are general dataframe transforms, not statistics

## [0.2.0] — 2026-06-23

### Added
- `--version` flag (`dfstat --version` prints `dfstat 0.2.0`)
- `__version__` exposed in `stattools.__init__` via `importlib.metadata`
- `dfstat help` now groups commands by category (Data transformation, Statistics, Plots, Utilities)
- New subcommands: `concat`, `describe`, `randvar`, `sample`
- New subcommand: `annotate` — read/write provenance metadata in parquet files
- New subcommand: `dataset` — load example datasets from seaborn/statsmodels
- New subcommand: `test` — p-values between column pairs (t-test, Mann-Whitney, bootstrap, …)
- Parquet pipe system: `-o` writes a temp parquet; `...` reads a parquet path from stdin
- Provenance metadata propagates automatically through multi-step pipelines via parquet schema

### Changed
- Installation method updated to `uv tool install` (no venv activation needed)
- `dfstat help` output grouped with subheaders instead of a flat list
- `--backend` option simplified to `pandas` / `duckdb` (polars removed)
- `randvar`: uses `common.seed.normalize_seed` instead of a private duplicate; `name`/`help` converted to `@property`; `--list` routes through `io.printdf`
- `concat`: added missing read options (`--backend`, `--nrows`, `--readasobject`, `--prequery`)
- `describe`: fixed `all_missing`/`high_missing` flag ordering; `--correlations` now requires `--summary`
- Git workflow established: feature branches for multi-file changes, conventional commit prefixes (`feat:`, `fix:`, `test:`, `chore:`, `docs:`, `refactor:`)

### Fixed
- matplotlib `font_manager` INFO messages suppressed on WSL2
- Ruff lint violations across all source files (E501, B904, B023, B905, E741, F841, E402, E701)

### Tooling
- ruff + ruff-format added (pre-commit hooks)
- pytest-cov added; 644 tests at 69% coverage
- GitHub Actions CI workflow (push/PR to main): lint + test
- Two Claude subagents added: `stattools-code-reviewer`, `stattools-test-writer`

---

## [0.1.0] — 2026-06-21

### Added
Initial release. Core subcommands ported from the legacy `df` package:

**Data transformation:** `eval`, `query`, `merge`, `melt`, `pivot`, `func`, `scale`, `interp`

**Statistics:** `stat`, `fit`

**Plots:** `scat`, `line`, `hist`

**Utilities:** `print`, `clean`, `help`

- Consistent CLI interface across all commands (`-o` pipe mode, `--select`, `--drop`, `--postquery`, `--meta`, etc.)
- TSV stdin/stdout as primary I/O; parquet as intermediate format for large pipelines
- Grouping (`-g`) and subplot grids (`--subgraphcol`) supported across plot commands
- Publication-quality figure presets (`--size`, `--fontsize`)

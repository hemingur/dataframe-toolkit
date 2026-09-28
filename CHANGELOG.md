# Changelog

## [Unreleased]

### Fixed
- `dftk ... | head` exited with status 120 (sometimes with an "Exception ignored while flushing sys.stdout: BrokenPipeError" message) once output exceeded the pipe buffer. dftk now exits quietly with status 141, like `cat`/`grep`. The old per-writer workaround also swallowed genuine write errors on named `-o` files; those now surface.
- `--backend duckdb` failed on every named TSV/CSV file (`query(): incompatible function arguments`). It now reads via `duckdb.read_csv`, and also honours `--noheader`.

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

# Changelog

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

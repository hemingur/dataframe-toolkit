# Changelog

## [Unreleased]

### Added
- `manhattan --miami COL [UPPER LOWER]`: two scans in one plot, the second mirrored below the axis (a Miami plot). COL holds two values, the first in the table drawn above, or name the two to draw (other rows are left out). The threshold lines are drawn on both sides, the tick labels have no minus sign, and the scan names are written in the corners. `--ylim -30 40` caps the lower half at 30 (triangles pointing down) and the upper at 40. Works with `--subgraphcol`, not with `-g`. The default size is 12x6 instead of 12x4.
- `manhattan --logy [T]` and `qq --logy [T]`: the y axis is linear up to -log10(p) = T (default 10) and logarithmic above it, so a peak at 300 and one at 8 both show. Ticks at 1, 2 and 5 times each power of ten, or only the powers on a tall axis.
- `manhattan --sigcolor COLOUR`: points past the `--sig` line in that colour.

## [0.19.0] — 2026-11-12

### Added
- `qq`: QQ plot of p-values, observed against expected -log10(p) with the y = x line, a pointwise confidence band (`--ci`, default 0.95, from the Beta distribution of each rank; `--no-ci`) and the genomic inflation factor λ (median chi-squared over its null median) on the plot and in the log (`--no-lambda`). The p-values come as in `manhattan`: `Pval`, `--pcol`, `--mlogpcol` or `--chi2col` with `--chi2df` (λ on that df). `-g` gives one curve and λ per group (studies, allele-frequency bins), `--subgraphcol` a panel each; `--ylim` caps the axis with triangles on the top edge; points with p > 0.01 are thinned (`--no-thin`). Ten million p-values take about 4 s.

### Changed
- `dataset gwas` has 120,899 variants instead of 43,604: 120k null background variants and 150 instead of 600 around each peak, so the peaks are about 1 % of the rows and λ is 1.02 as in a real scan (it was 1.17). The six peaks are unchanged.

## [0.18.0] — 2026-11-12

### Added
- `manhattan`: Manhattan plot, -log10(p) along the genome. Columns `Chr`, `pos` and `Pval` by default (common spellings such as `CHR`, `BP`, `P` found ignoring case), or `--mlogpcol` (-log10(p)) or `--chi2col` with `--chi2df` (exact where the p-value underflows to 0). Chromosomes in genome order with alternating colours (`--colors`), `--sig` (5e-8) and `--suggestive` (1e-5) lines, `none` to hide; `--ylim` caps the axis, with points above it as triangles on the top edge. Points with p > 0.01 are thinned to a fine grid (`--no-thin`) and rasterized: ten million variants take about 6 s and a PDF about 200 KB. `-g` colours several scans (with grey bands telling the chromosomes apart), `--subgraphcol` gives each a panel.
- A two-page cheatsheet, `docs/cheatsheet/dftk-cheatsheet.pdf`, linked from the README: basics, reading and writing, inspecting, new columns, SQL with DuckDB, reshaping, statistics, plots and pipeline recipes. `docs/cheatsheet/make_cheatsheet.py` rebuilds it with pdflatex after running every example against the installed dftk and drawing the plot thumbnails with dftk.
- `dataset gwas`: a synthetic association scan (about 44k variants on chromosomes 1–22 and X, six peaks) with columns `SNP`, `Chr`, `pos`, `Pval`, `Chisq` (1 df) and `r2` (LD with the nearest peak's causal variant, within 1 Mb), for the coming genome-wide plots. dftk makes it itself, with no download, and `dataset` looks in dftk's own datasets first.
- `eval -f` statistical functions take numbers as arguments: `chi2_to_neglogp(Chisq, 1)`, `binom_test(k, n, 0.5)`, no `-c 'k = 1'` column needed first. A column with that name still wins.

### Changed
- `eval -f` runs `chi2_to_neglogp`, `neglogp_to_chi2`, `pval_to_chi2`, `t2pval` and `pval2se` on whole columns instead of row by row: a million rows take about 1.5 s instead of about 2 minutes. The exact tests (`fisher_test`, `boschloo_test`, `binom_test`, ...) still run row by row. A wrong number of arguments is an error naming them (`chi2_to_neglogp() takes 2 arguments (chi2, df), not 1`).

### Fixed
- `eval` `chi2_to_neglogp` is exact for very large chi-squared values (p below about 1e-300). It switched to an approximation at chi-squared 1380, before scipy needed it, and the approximation kept one term of its series: -log10(p) was off by 0.0003 at df 1 and 0.07 at df 200. scipy is now used for as long as it gives a value, and the series beyond that is cut at its smallest term; checked against 60-digit mpmath to about 1e-15 for df 1 to 1000 and chi-squared up to 1e7.
- `eval` `neglogp_to_chi2` is the exact inverse of `chi2_to_neglogp` above -log10(p) = 300; its one-term approximation was off by up to 4e-5 (relative) there.

## [0.17.1] — 2026-11-10

### Fixed
- `--sortasc` or `--sortdesc` without column names did nothing (in every command, also in the legacy `df`). It now sorts by every column, left to right, like the shell's `sort`; on a one-column file that is the column itself.
- A failed sort is an error: a misspelled column (`--sortasc colx`) printed a warning and the table unsorted, with exit code 0, so a script carried on. The error names the missing column and, when the name is a file (`dftk print --sortasc data.tsv` reads the file as a column), says to put the file first. Text and numbers in one sort column are an error too.
- `--sort` takes only `COL:a` and `COL:d`; another direction (`COL:asc`, `COL:x`) was read as ascending. A column name may contain `:` (`chr:pos:a`).

## [0.17.0] — 2026-11-09

### Added
- `append`: add one row from `-c COL=VALUE` values to a table (DATAFILE or stdin); columns not given are empty (`--fill`), a new column goes at the end. Without input it makes a one-row table, e.g. a small JSON record with `--format json`. Values: numbers and `true`/`false` are typed, a quoted value stays text, an empty one is missing; a value keeps its column's type where it can.
- `append --inplace FILE`: add the row to the file itself, creating it if missing, for status logs. Parallel writers are safe (the file's directory is locked). A TSV row whose columns are all in the header, and any `.jsonl` row, is one more line at the end; a new column or a `.parquet` file rewrites the file through a temp file, leaving existing TSV cells exactly as written. `--timestamp [COL]` adds the local time as ISO 8601 text (`2026-11-09T19:13:57`; a timestamp type in parquet); `--strict` refuses columns the file doesn't have.

### Fixed
- `eval -c` with a quoted value (`-c 'code = "007"'`) kept the quotes in the value; it is now the text inside them.

## [0.16.0] — 2026-11-07

### Added
- `heatmap`: a coloured grid of values, from a long table (`-x COL -y COL -v VALUE`, one row per cell, as `corr` and `stat` give it) or a wide one (`-y ROWS`, the other numeric columns as heatmap columns, as `pivot` gives it). `--annot` writes values in the cells (`--fmt`), `--cmap`, `--center`, `--vmin`/`--vmax`, `--lower` (lower triangle), `--square`, and the figure options of the other plots. Values of both signs get diverging colours centred on 0.
- `pairs`: scatter-plot matrix, every pair of columns (`-c COL…`, globs; default all numeric) as a small scatter plot with each column's KDE on the diagonal (`--diag hist` for histograms). `-g` colours groups with one KDE per group, each scaled by the group's share of the rows; `--upper corr` writes the correlation (`--method`) in the upper triangle on a cell coloured by its value, or with `-g` r over all rows and r within each group (they can differ even in sign). Points are rasterized so a PDF stays small; `--alpha` and `--markersize` for large tables.
- `corr --all [COL…]`: the correlation matrix, every pair both ways round plus the diagonal, in column order, for a correlogram: `corr --all -o | heatmap -x col1 -y col2 -v correlation --annot --vmin -1 --vmax 1`. Each pair is computed once; on the diagonal r is 1 and the p-value, intervals, bootstrap and tail columns are empty. A pair with fewer than 2 complete rows gets empty statistics and a warning naming it.
- `corr -c` takes globs on either side, expanded to numeric columns: `-c '*:*'` gives every pair once, `-c 'sepal*:petal*'` the cross product. A column whose name contains `* ? [` (`x[1]`) still matches only itself.

## [0.15.1] — 2026-11-07

### Fixed
- `.tex` output: a row whose first cell starts with `[` (a JSON list, say) no longer breaks the table; LaTeX read `\\[` as a line-spacing argument. `[` and `]` are now written as `{[}` and `{]}`.
- TSV, Markdown and LaTeX output crashed on repeated column names (e.g. after `--move id:s` when `s` exists); it worked before 0.15.0. JSON output, which can't hold a repeated key, now says so instead of keeping only the last column.
- `NaN` and `Inf` inside lists are written as `null`, so JSON output and the JSON text in TSV are valid JSON.
- `merge` keeps the JSON columns from `query --sql` (fields that are a string in some records and a list in others): JSON output after a merge writes them as JSON again, not as quoted text.

## [0.15.0] — 2026-11-06

### Added
- Output formats for documents and JSON tools: `-o FILE.json` (array of records), `.jsonl` (JSON Lines), `.md` (Markdown table, numeric columns right-aligned) and `.tex` (bare booktabs `tabular` for `\input`, LaTeX special characters escaped, `| < >` too; use `\usepackage[T1]{fontenc}` for straight `"`). `--format tsv|json|jsonl|md|tex` writes them to stdout or overrides the extension. `--digits` applies. Replaces the legacy `df --latex`; no new dependency.
- JSON output keeps nested values: lists and structs from `query --sql` are written as JSON again, and so are fields that are a string in some records and a list in others (DuckDB type JSON), also after an `-o` parquet pipe. A key missing from a record is written as `null`.

### Fixed
- TSV output of lists and nested values: `query --sql "SELECT * FROM 'catalog.json'"` crashed ("need to escape, but no escapechar set") when a list was long enough for numpy to wrap it over several lines, and shorter lists were printed numpy-style (`['a' 'b']`). They are now one line of JSON (`["a","b"]`). Plain strings in a DuckDB JSON column are shown without their JSON quotes.

### Documentation
- README "Output formats"; README "JSON files" shows `json_type()` for mixed fields and JSON output; `query -h`, `print -h` and the skill updated.

## [0.14.0] — 2026-11-05

### Changed
- `query --sql` reads no input when the SQL doesn't use the `data` table (or `--table NAME`), so `dftk query --sql "SELECT * FROM 'f.jsonl'"` runs on its own. Before, it waited for stdin, or failed with "Empty CSV file" in scripts. A named DATAFILE is still read.

### Documentation
- README "Files without a header": merging two headerless files (`V1 V2 V3 V2_r V3_r`) and naming the merged columns with `--move`.
- JSON files: `query -h` and README "JSON files" show reading JSON and JSON Lines with DuckDB, and flattening nested objects (`pos.*`) and lists (`unnest`, `len`).

## [0.13.0] — 2026-11-03

### Changed
- `--noheader` now only describes the input. TSV output gets a header (`V1 V2 …`, or the names given with `--move`) unless `--removeheader` is given. Before, `--noheader` also left the header out of the output, so tables that commands build (`stat`, `pivot`, …) lost their column names and `--move` names were silently dropped. **Scripts that relied on headerless output: add `--removeheader`.**
- `print` takes the standard output options (`--move`, `--select`, `--removeheader`, `-o FILE`, …), so `dftk print data.txt --noheader --move V1:chr V2:begin V3:end` names the columns of a headerless file.
- `--removeheader` with parquet output (`-o`, `-o x.parquet`) and `--noheader` with parquet input now warn that they have no effect (parquet always has column names).

### Fixed
- `print` wrote `""` for a missing value in a one-column table; it now writes an empty line, like every other command.

### Documentation
- README: "Files without a header" (`--noheader`, `--move`, `--removeheader`).
- `eval -h`: a regular expression test, `x.str.fullmatch('[A-Za-z]{7}')` (True when the whole value is 7 ASCII letters; `match` only checks the start).

## [0.12.0] — 2026-11-01

### Added
- `dftk skill install`: installs the agent skill that now ships with dftk into `~/.claude/skills/dftk` and/or `~/.codex/skills/dftk` (whichever agent is found), or `--claude`, `--codex`, `--dir DIR`. It prints where it installed it, and replaces only an earlier copy of the dftk skill. `dftk skill path` prints the packaged folder. Run it again after upgrading dftk. The skill moved from `skills/dftk/` to `src/dftk/skills/dftk/` so it is in the wheel.
- Reading stdin no longer needs the `...` marker: when stdin (no DATAFILE, or `-`) holds just the path of an existing `.parquet` file, as printed by `-o`, that file is read; anything else is read as TSV, as before. `dftk a -o | dftk b -c x` now works in every shell. zsh with Oh My Zsh turns an unquoted `...` into `../..`, so scripts using it didn't run there. `...` still works.
- The same applies to a pipe given as a file: `<(dftk a -o)` used to fail with "Empty CSV file".
- `merge`: the right side can come from stdin. `-l left.tsv -r -`, or `-l left.tsv` with `-r` left out, reads the right side from stdin.

### Changed
- `merge -l FILE -r -` used to join FILE with itself (with a warning); `-r -` now reads stdin. With stdin on the left (`-l -`, or no `-l`), `-r -` is still a self-join.

### Documentation
- Help, README and the agent skill no longer use the `...` marker: examples are `dftk a -o | dftk b args`, which works in every shell. The README's Input/output section explains stdin detection and that `...` still works; CLAUDE.md's help-example rule is updated to match.

### Fixed
- `split` stopped with a traceback when the folder in `-p`/`--template` didn't exist (`-p out/`). Missing folders are now created, also nested ones (`--template out/{day}/{time}.tsv`).
- A directory as DATAFILE (what an expanded `...` becomes) crashed with a traceback. It's now a one-line error that names the likely cause.
- The pandas reader read a one-column file given as a pipe (`<(...)`) twice, so empty lines (missing values) were dropped.

## [0.11.3] — 2026-10-27

### Security
- Requires `pillow>=13.0` (used by matplotlib to write plot images), which fixes the security advisories against pillow up to 12.3.0. dftk doesn't use pillow directly; the floor makes an upgrade of dftk also upgrade a pillow already installed.
- Lock file: pillow 13.0.0, and for development only (via pre-commit) filelock 4.2.4 and virtualenv 21.15.0.

### Documentation
- `query -h`, README and the agent skill show how to compare a row with the previous or next one with `LAG()`/`LEAD()`, e.g. the gap between segments (`"Begin" - LAG("End") OVER (PARTITION BY Chr ORDER BY "Begin")`), and that keyword column names such as `Begin`/`End` need double quotes. README also shows the `eval -f` form with `shift`.
- `eval -h` and README: differences within a column, `x.diff()` and `x.pct_change()`, also per group (`x.groupby(g).diff()`), and that the rows must be sorted first.

## [0.11.2] — 2026-10-25

### Fixed
- `eval -f`: a dftk function inside a larger expression, e.g. `z = sum(a, b) * 2`, silently ignored the rest (`* 2`). A dftk function must be the whole right-hand side; such a formula is now an error with a two-step example (`-f "tmp = sum(a, b)" -f "z = tmp * 2"`).

### Changed
- `eval -f` decides up front whether a formula is a call to a dftk function or a pandas expression, instead of trying pandas first and falling back. Results are the same; error messages are simpler (a pandas syntax error no longer says "Expected: dest = func(...)", and an unknown function gets one line).
- `eval -s`: `exists(path)` and `isfile(path)` return True/False instead of 1/0, like `-f`'s `exists`. Use `--intbool` for 1/0.
- `eval -h` is shorter: the "Filling missing values" section is a table of the fill methods, the sorting and grouping rules and one example. The full explanation, with defaults from a merge and interpolating within groups, is in the README.
- Internal: eval's function registry moved from `commands/eval_cmd.py` to `common/evalfuncs.py`.

## [0.11.1] — 2026-10-25

### Fixed
- `eval -f`: a formula without `name =` (`-f "a + b"`) crashed with a traceback since 0.11.0. It's now an input error that shows how to write it.
- `eval -f`: a special function given a column that doesn't exist (`sum(a b)`) or a number (`binom_test(k, n, 0.5)`) crashed with a `KeyError`. It's now an input error naming the column; for a number it suggests adding it as a column first (`-c "p = 0.5" -f "z = binom_test(k, n, p)"`).
- `eval -f` on an empty table skipped every formula error and just added the column, so a typo went unnoticed until there was data. An empty table has no column types, so errors that depend on them are still skipped, but unknown columns, methods and functions, bad syntax and a missing `name =` are reported.
- `implode`: a number column containing `inf` crashed, and a column named `_gid` in the input was overwritten by dftk's internal group number.

## [0.11.0] — 2026-10-24

### Fixed
- `eval -f`: assigning a result of another type to an existing column failed, e.g. `depth = depth.fillna(depth_r)` after a merge (a float column filled from a nullable integer one) or `x = x.isna()`, with a misleading "not available as a special function" error. The column now takes the result's type; no data is copied.
- `eval -f`: a mistake in the expression (an unknown column or method, wrong arguments, `.str` on numbers) came out as "Unexpected error … please report" with a traceback. It's now a short input error that names the formula; for text methods it suggests `.str` (`species.lower()` → `species.str.lower()`). When the special-function fallback also fails, pandas' own error is included.

### Added
- `dftk implode`: the inverse of `explode`. Joins a column's values within groups of rows into one cell (`GO:1,GO:2`). Rows are grouped by all the other columns, or by `-i` (the other columns are then dropped); groups keep the order of their first row, and rows with an empty key form their own group. `--sep` (default `,`). Several `-c` columns are joined in the same row order; `--poscol NAME` puts the values back in the order of `NAME` (e.g. from `explode --poscol`) and drops it. `--unique` drops repeats and `--sortvalues` sorts the values (with several columns, both work on whole rows). Empty values are left out, and whole-number columns read as floats are written without `.0`.
- `eval -h` has a "Filling missing values" section: `fillna` from another column, a fixed value or a group mean, defaults from a merge, `ffill`/`bfill`/`interpolate`, filling within groups with `groupby`, and why rows must be sorted first. The README has a short version.

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

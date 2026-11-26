# Changelog

## [Unreleased]

### Added
- `scat --abline` and `line --abline A:B[:LABEL]`: dashed lines y = A + B*x (intercept, then slope, as in R's `abline`), in the style of `--vline`/`--hline`, labelled along the line when `:LABEL` is given. A bare `--abline` is the identity line y = x. The line spans the whole panel and leaves the axis limits as they are; on log axes it is drawn as the curve it is. Repeatable, so `--abline --abline=-1:2` draws both (a negative intercept needs `=`). With `--subgraphcol` the same lines go on every panel.
- `--cast COL:str` (or `string`) makes a text column, written to parquet as a string column: 1 becomes "1", 2.2 "2.2", and empty cells stay empty. Casts apply in order, so `--cast c:Int64 c:str` turns 1.0 into "1". Useful for numeric codes (cluster number, class label) that should be categories.

### Changed
- A `--cast` that can't be done is now an input error (`Input error: …`, exit 1) that says why: an unknown type (it lists the valid ones), a column name that matches nothing (a glob matching nothing is still fine), a malformed `COL:TYPE`, or a conversion that fails, e.g. 2.5 to `Int64`. It used to log `--cast: failed for …` with no reason and write the column unchanged. A pipeline that relied on a failing `--cast` being skipped now stops.
- `--cast COL:int` (or `Int64`) on numbers that aren't whole is an error that says to use `--round COL:0` first. `int` used to truncate them silently (2.6 to 2, -1.7 to -1).
- A malformed `--round` (`v:x`), a column it names that doesn't exist or isn't numeric, is an input error; it used to be skipped without a message. A glob (`--round 'x*:2'`) rounds the numeric columns it matches and skips the rest.

### Fixed
- An invalid `--postquery` or `--prequery` expression (an unknown column, a syntax error) is an input error, one `Input error: --postquery 'x > 1': name 'x' is not defined` line and exit 1, instead of an "Unexpected error" with a pandas traceback. Commands from other packages that use dftk's output options get the same.
- `--round` is done before `--cast`, so `--round x:0 --cast x:int` gives the nearest integer (2.6 to 3, -1.7 to -2). The cast used to come first: `int` truncated (2.6 to 2), and `Int64` failed with a warning and left the rounded floats (3.0).

## [0.28.0] — 2026-11-24

### Changed
- `--size` on every plot is the size of the whole figure, exactly, with the `--subgraphcol` panels sharing it. It used to be the size of one panel, multiplied by the grid: `--size double` (a journal's 7.2" column) with 8 panels in 4 columns gave a figure 28.8" wide, and a `WxH` size was silently capped at 16x10, even `--size 20x12` for one panel. Without `--size` nothing changes: each panel gets the command's default size and the figure grows with the panels up to 16x10. `pairs --size 10x6` now gives 10x6 too (it used to be taken for the default and ignored).

### Fixed
- `heatmap --annot`: the numbers in the cells are as large as the tick labels and follow `--fontsize`. They were always 10 pt, larger than the 9 pt tick labels by default and nearly twice the 6 pt ones with `--fontsize publication`, which crowded grids of many cells or panels. `--annotsize PT` sets them alone.

## [0.27.0] — 2026-11-24

### Added
- `phewas --labelpercat K`: at most K labels per category (per panel), strongest first, so several strong hits in one category (variations of one phenotype) no longer take every label; `--label N` still caps the total. The log says how many hits above the line were left without a label.
- `phewas --labelnumbers`: labels are numbers, 1, 2, … left to right, instead of phenotype names, and the key (number, category, phenotype, Pval, and the `--betacol` and `--subgraphcol` values) goes to stdout as TSV for a caption (`> key.tsv`). With `--subgraphcol` a phenotype keeps one number in every panel.

### Changed
- `phewas`: every category is now as wide on the x axis, its phenotypes spread evenly across it, instead of one unit per phenotype (a category with 4 phenotypes was half as wide as one with 9). `--catwidth count` gives the old layout.

### Documentation
- JSON files: README "JSON files", `query -h` and the skill say that dftk reads JSON as a table, so writing it back as JSON gives every record every key (`null` where a record had none), and point to `jq` for filtering JSON that should stay JSON (`jq -c 'select(...)'`) or turning it into TSV for a pipeline (`jq -r '[...] | @tsv'`).

## [0.26.0] — 2026-11-19

### Added
- `phewas --subgraphcol COL`: one PheWAS per value of a column (burden models of one gene, genes within one model, genotypes), stacked one per row by default (`--ncols` for a grid, `--subgraphorder`, `--subgraphformat`). A phenotype stands at the same x in every panel, one missing from a panel leaves a gap, and all panels share one y axis. The Bonferroni line counts each panel's own tests, and the log gives each panel's threshold when they differ; `--sig P` draws one line at P in every panel. Hits are labelled per panel; the `--betacol` legend and the category names appear once.

## [0.25.1] — 2026-11-19

### Fixed
- `heatmap --subgraphcol`: when the last row of panels is short (8 panels in 3 columns), the rows are evenly spaced again. The tick labels of a panel above an empty grid cell used to widen the gap under its whole row.

## [0.25.0] — 2026-11-18

### Added
- `heatmap --subgraphcol COL`: one heatmap per value of a column, side by side (`--ncols`, `--subgraphorder`, `--subgraphformat`). Every panel has the rows and columns of the whole table in the same places, and all share one colour scale and one colour bar (`--vmin`, `--vmax`, `--center` apply to all), so a colour means the same value in every panel. Tick labels only on the outer edge. In wide input (`-y` alone) the subgraph column is never a heatmap column, even when numeric.
- `box --hline Y[:LABEL]`: dashed reference lines across the boxes, as on `scat`, `line` and `hist`, e.g. a detection limit or a population mean (`--hline 3:mean`, example in `box -h`); also on a `--logy` axis. The label has a light background so it can be read over a box.

## [0.24.0] — 2026-11-17

### Added
- `box`: box plot of a value (`-y`) by category (`-x`; without it, one box of all rows), categories in sorted order (numbers numerically, so a genotype 0/1/2 is in order) unless `--order` gives one. Tukey whiskers; `--kind violin` draws densities instead. `-g` splits each box into one per group, side by side and in colour (`--groupcolorder`). `--points` draws every point on top, jittered; `--count` writes the number of values under each category; `--logy`, `--ylim`. Rows without a value or category are left out.
- `phewas`: PheWAS plot, -log10(p) of each phenotype with the phenotypes grouped by category along the x axis, one colour per category, categories in order of first appearance. Columns `category`, `phenotype` and `Pval` unless named (`-c`, `-l`; `--pcol`, `--mlogpcol`, `--chi2col` as in `manhattan`). `--betacol` draws triangles pointing up or down with the sign of the effect, with a legend. The significance line is at 0.05 / number of tests (Bonferroni) unless `--sig P` (or `none`); `--suggestive` adds a second. The phenotypes above the line are labelled, strongest first, at most `--label N` (default 10), moved apart where they would overlap. `--ylim` caps points on the top edge; `--logy` as in `manhattan`.
- `scat --xerr COL --yerr COL`: error bars from columns of ± values (e.g. standard errors), in each point's colour, under the points; `--errscale Z` multiplies them (1.96 for 95 % intervals). Not with `--hexbin`.
- `scat -w/--weights COL`: weighted least squares for the `--fit ols` line; rows without a weight are left out and the legend says "(weighted)". With `--nointercept` and weights 1/se² this is the inverse-variance-weighted (IVW) Mendelian-randomisation slope, drawn over its instruments with `--xerr`/`--yerr` (example in `scat -h`). Not with `--robust` or `--fit lowess`.
- `--vline X[:LABEL]` and `--hline Y[:LABEL]` on `scat`, `line` and `hist` (and `--vline` on `forest`): dashed grey reference lines, several per option, each labelled along the line when written `X:LABEL`, e.g. the carrier count on a power curve (`--vline 412:cohort`, example in `line -h`). They are kept in view unless `--xlim`/`--ylim` leave them out.

## [0.23.0] — 2026-11-17

### Added
- `forest`: forest plot, one row per estimate with its confidence interval, labels down the left in file order. The interval comes from two columns (`--ci LO,HI`) or a standard error (`--se SE`, ± z·SE for `--level`, default 0.95). `--exp` draws exp() of estimate and interval on a log axis with the reference line at 1 (log odds ratios become odds ratios); `--logx` for ratios already in the table; `--ref` moves or removes the reference line. `--text` writes `estimate [lo, hi]` in a column to the right (`--digits`), and `--textcol` adds columns as they are (p-values, carrier counts); the figure grows to make room. `-g` draws several series per row side by side, in colour (`--groupcolorder`), with the text lines in the same colours; `--section` puts rows under bold headers. `--weighted` sizes markers by 1/SE²; `--xlim` ends intervals that run off the axis in an arrow. Works with `fit` output directly (`-y marker -x effect --ci cilo,cihi`).
- `dataset phewas`: a simulated phenome-wide scan of loss-of-function carriers to try `forest` and `qq` on: category, phenotype, prevalence, genotype (het, hom), n_carriers, beta (log odds ratio), se, Pval for 74 phenotypes. Five have a real effect (one protective), significant in the 4,000 heterozygotes; the 60 homozygotes give wide intervals.

## [0.22.1] — 2026-11-16

### Fixed
- `line --yci LO,HI` works again when the table comes through an `-o` pipe (or any parquet file): since 0.20.0 the two columns weren't read, and the example in `line -h` failed. A `--yci` value that isn't two comma-separated names is a clean error.
- `stat` gave empty `cilo` and `cihi` for any column or group with a missing value; they now leave missing values out, like the other statistics. `stat -h` and `line -h` say what they are: percentiles of the data (the middle 95 % by default), not a confidence interval of the mean.
- `eval` `where()`:
  - an argument that isn't a column, a quoted string or a number is now evaluated as a pandas expression, so `where(tip > 3, -tip, tip)` works in one step. Before, `-tip` became the text "-tip" and a condition like `tip > 3` crashed with a traceback;
  - an unquoted word that is not a column is an error that says to quote it, instead of silently becoming text;
  - the condition must be true/false (or 0 and 1; missing counts as false): a text column is an error suggesting a comparison, instead of every row counting as true;
  - commas inside quotes or calls no longer split arguments, and text mixed with numbers (`where(big, "large", 0)`) no longer fails.
- Help examples split over lines with a trailing backslash printed as one long line in `eval -h`, `scat -h` and `manhattan -h`; a test now checks every command's help.

### Added
- `line -ls/--linestyle STYLE`: `-ls none -m o` draws points with their error bars and no connecting line (a dot-and-whisker plot).

## [0.22.0] — 2026-11-15

### Added
- `locus`: regional association plot, the zoomed-in companion to `manhattan`. -log10(p) against position in Mb for the rows it is given (cut the region out with `query`; input on several chromosomes is an error that suggests a query 500 kb either side of the top hit). Columns and p-value options as in `manhattan`. `--ldcol r2` colours the points by r² with the lead variant in five bins (0–0.2 … 0.8–1), grey where missing, most LD drawn on top, with a legend. The lead variant is a diamond, the smallest p-value unless `--lead` names one; `--label` writes more names on the plot, with a ring round each labelled point. Variants are named by ID (`SNP` column, found by alias such as `rsid`, or `--idcol`) or by position (`chr6:32600000`, `6:32600000`); several rows at one position take the smallest p. `--sig` (default 5e-8), `--suggestive` (default none), `--ylim`, `--logy` and thinning as in `manhattan`. dftk doesn't compute LD; r² is merged in. No gene or recombination track yet. Fixes #37.

## [0.21.0] — 2026-11-15

### Added
- `func -t interpolate`, `-t ffill` and `-t bfill` fill empty cells within groups (`-g`). They require `-x/--xcol X`: each group's rows are taken in X order whatever order the file is in (no sort step), the output keeps the file's row order, and `interpolate` draws the line against X: unevenly spaced rows get the right value (10 at x=2 and 14 at x=6 give 13 at x=5; `eval`'s `interpolate()` counts rows and gives 12). Gaps before a group's first value or after its last stay empty. X can be numbers or dates for `interpolate` (which needs a numeric column, and returns decimals), and anything sortable for `ffill`/`bfill` (numbers, dates, text; rows with the same X keep their file order). Repeated X values within a group are an error for `interpolate`, naming the group; rows without an X value are left alone. Chaining `interpolate`, `ffill` and `bfill` on the same column leaves no gaps. A million rows in 10,000 groups take about half a second. Fixes #15.

### Changed
- `eval -h` and the README point to `func -t interpolate` for interpolating within groups; the README's masked `eval` recipe for it is gone.

## [0.20.0] — 2026-11-15

### Added
- `manhattan --miami COL [UPPER LOWER]`: two scans in one plot, the second upside down under the first (a Miami plot). COL holds two values, the first in the table drawn above, or name the two to draw (other rows are left out). The lower scan is drawn upside down under the upper one, on the same scale, with the chromosome numbers in a gap between them; each half has its threshold lines and its scan's name in the corner. `--ylim -30 40` caps the lower half at 30 (triangles pointing down) and the upper at 40. Works with `--subgraphcol`, not with `-g`. The default size is 12x6 instead of 12x4.
- `manhattan --logy [T]` and `qq --logy [T]`: the y axis is linear up to -log10(p) = T (default 10) and logarithmic above it, so a peak at 300 and one at 8 both show. Ticks at 1, 2 and 5 times each power of ten, or only the powers on a tall axis. Points are thinned on the axis as drawn, so the stretched linear part of a small T is as dense as without thinning.
- `manhattan --sigcolor COLOUR`: points past the `--sig` line in that colour.

### Changed
- `manhattan`, `qq`, `scat`, `line`, `hist` and `chi` load only the parquet columns they use, from a named file or a `-o` pipe. On a 10-million-row, 13-column regenie-style file, `manhattan` takes 12.5 s and 1.6 GB instead of 16.5 s and 3.5 GB, and `qq` 1.0 GB instead of 2.7 GB; the plots are byte-identical. `manhattan` and `qq` find their columns (`CHR`, `P`, `LOG10P`, …) from the file's schema first. With `--prequery`, which can use any column, every column is still read; TSV input is unchanged.

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

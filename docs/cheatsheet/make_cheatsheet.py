# ruff: noqa: E501  (the cheatsheet text is kept as long strings)
"""
Build the dftk cheatsheet: two landscape A4 pages, three columns each.

    python docs/cheatsheet/make_cheatsheet.py            # check, draw, build
    python docs/cheatsheet/make_cheatsheet.py --no-check # skip running examples

Every example is run against the installed dftk first (a failing one stops
the build), the thumbnails are drawn with dftk itself, and pdflatex builds
dftk-cheatsheet.pdf from the generated cheatsheet.tex.

Examples are shown without their input; the check pipes in the dataset given
with each entry (dftk dataset NAME -o | ...), or runs the entry's own shell
command in a folder with a few small files (FILES below).
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Small files for the examples that name a file
FILES = {
    "data.tsv": "id\tgrp\tx\ty\n1\ta\t1.5\t2.0\n2\tb\t2.5\t3.5\n3\ta\t0.5\t1.0\n",
    "left.tsv": "id\tname\n1\tAda\n2\tBo\n3\tCy\n",
    "right.tsv": "id\tscore\n1\t10\n2\t20\n4\t40\n",
    "a.tsv": "id\tx\n1\t1.5\n",
    "b.tsv": "id\tx\ty\n2\t2.5\t7\n",
    "curve.tsv": "x\ty\n0\t0\n10\t100\n",
    "samples.tsv": "x\n2.5\n7\n",
    "labels.tsv": "day\tkind\nThur\tweekday\nFri\tweekday\nSat\tweekend\nSun\tweekend\n",
    "f.jsonl": '{"id": 1, "pos": {"chr": "1", "start": 100}, "tags": ["a", "b"]}\n',
    "f.txt": "Name\tUpph\xe6\xf0\nAda\t1,500.00\n".encode("latin-1"),
}

# ---------------------------------------------------------------------------
# Content.  A section is (title, colour, items); an item is one of
#   ("cmd", code, description, check)  check: dataset name, "sh:COMMAND",
#                                      or None (not run, e.g. a pattern)
#   ("sub", heading)
#   ("text", latex)
#   ("img", [png, ...])
#   ("block", code lines, check)       a multi-line pipeline
# ---------------------------------------------------------------------------

BASICS = (
    "Basics",
    "c1",
    [
        (
            "cmd",
            "dftk CMD [FILE] [options]",
            "every command reads TSV or parquet and writes TSV to stdout",
            None,
        ),
        (
            "cmd",
            "cat data.tsv | dftk stat -c x",
            "no FILE: read stdin",
            "sh:cat data.tsv | dftk stat -c x",
        ),
        (
            "cmd",
            "dftk dataset tips -o | dftk stat -c tip",
            "\\texttt{-o} alone writes a temp parquet and prints its path; the next command reads it (types kept, fast, deleted after use)",
            "sh:dftk dataset tips -o | dftk stat -c tip",
        ),
        (
            "cmd",
            "dftk CMD -h",
            "options and examples, ready to run",
            "sh:dftk pivot -h",
        ),
        (
            "cmd",
            "dftk dataset --list",
            "about 50 example tables: tips, iris, penguins, diamonds, \\ldots",
            "sh:dftk dataset --list",
        ),
        ("cmd", "dftk skill install", "teach Claude Code and Codex to use dftk", None),
        (
            "text",
            "\\emph{Examples below leave out their input: run them as} \\texttt{dftk dataset tips -o | dftk \\ldots} \\emph{(or iris, penguins, diamonds, gwas, as the columns show).}",
        ),
    ],
)

READ = (
    "Read: options of every command",
    "c1",
    [
        (
            "cmd",
            '--prequery "tip > 5"',
            "filter rows as they are read",
            'sh:dftk dataset tips -o | dftk stat -c tip --prequery "tip > 5"',
        ),
        (
            "cmd",
            "--noheader",
            "no header row: columns are V1, V2, \\ldots",
            "sh:tail -n +2 data.tsv | dftk print --noheader",
        ),
        (
            "cmd",
            "--delimiter ,",
            "CSV or other separators",
            "sh:printf 'a,b\\n1,2\\n' | dftk print --delimiter ,",
        ),
        (
            "cmd",
            "--readasobject id",
            "keep a column as text (\\texttt{007} stays \\texttt{007})",
            "sh:printf 'id\\n007\\n' | dftk print --readasobject id",
        ),
        (
            "cmd",
            "grep -v '^#' f.tsv | dftk CMD",
            "drop comment lines first",
            "sh:printf '#c\\nx\\n1\\n' | grep -v '^#' | dftk print",
        ),
        (
            "cmd",
            "iconv -f latin1 -t utf-8 f.txt | dftk CMD",
            "Latin-1 file; \\texttt{sed -E 's/([0-9]),([0-9])/\\textbackslash 1\\textbackslash 2/g'} drops thousands commas",
            "sh:iconv -f latin1 -t utf-8 f.txt | sed -E 's/([0-9]),([0-9])/\\1\\2/g' | dftk cleannames",
        ),
        (
            "cmd",
            "dftk query --sql \"SELECT * FROM 'f.jsonl'\"",
            "JSON and JSON Lines through DuckDB (no input needed); \\texttt{pos.*} flattens, \\texttt{unnest(tags)} explodes",
            "sh:dftk query --sql \"SELECT * EXCLUDE (pos), pos.* FROM 'f.jsonl'\"",
        ),
    ],
)

WRITE = (
    "Write: options of every command",
    "c1",
    [
        (
            "cmd",
            "-o out.tsv",
            "to a file; \\texttt{.parquet}, \\texttt{.json}, \\texttt{.jsonl}, \\texttt{.md} (table) and \\texttt{.tex} (booktabs tabular) by extension",
            "sh:dftk print data.tsv -o out.md && cat out.md",
        ),
        (
            "cmd",
            "--format json",
            "the same formats to stdout: tsv json jsonl md tex",
            "sh:dftk print data.tsv --format json",
        ),
        (
            "cmd",
            "--select day 'tot*' +",
            "choose and order columns; globs; \\texttt{+} keeps the rest",
            "tips",
        ),
        ("cmd", "--drop size  --move tip:gratuity", "drop; rename (OLD:NEW)", "tips"),
        (
            "cmd",
            "--sortasc day tip",
            "sort; \\texttt{--sortdesc} descending; no names: every column",
            "tips",
        ),
        ("cmd", "--sort day:a tip:d", "mixed directions", "tips"),
        ("cmd", "--round tip:1  --digits 3", "decimals; significant digits", "tips"),
        ("cmd", "--cast size:Int64", "change a column's type", "tips"),
        (
            "cmd",
            "--deduplicate day  --dropna",
            "unique rows (by columns); drop rows with gaps",
            "tips",
        ),
        ("cmd", '--postquery "tip > 5"', "filter the output", "tips"),
        (
            "cmd",
            "--meta source=run7",
            "provenance stored in parquet output",
            "sh:dftk print data.tsv --meta source=run7 -o p.parquet && dftk annotate p.parquet",
        ),
    ],
)

INSPECT = (
    "Inspect and filter",
    "c2",
    [
        ("cmd", "dftk info", "types, missing values, memory per column", "tips"),
        ("cmd", "dftk describe", "profile with quality flags", "tips"),
        (
            "cmd",
            "dftk stat -c tip total_bill -g day",
            "n, mean, std, median, CI, sem, skew \\ldots per group",
            "tips",
        ),
        (
            "cmd",
            "dftk query -q \"tip > 5 and day == 'Sat'\"",
            "keep rows (pandas expression)",
            "tips",
        ),
        (
            "cmd",
            'dftk query --sql "SELECT day, avg(tip) AS tip FROM data GROUP BY day"',
            "any DuckDB SQL; the input is the table \\texttt{data}",
            "tips",
        ),
        (
            "cmd",
            "dftk sample -n 5 -g day --randomseed 1",
            "random rows, per group",
            "tips",
        ),
        (
            "cmd",
            "dftk split -g day -s .tsv",
            "one file per group: split-Sun.tsv \\ldots",
            "tips",
        ),
        (
            "cmd",
            "dftk cleannames",
            "\\texttt{Sample ID} $\\to$ \\texttt{sample\\_id}, \\texttt{GC \\%} $\\to$ \\texttt{gc\\_pct}",
            "sh:printf 'Sample ID\\tGC %%\\n1\\t2\\n' | dftk cleannames",
        ),
        (
            "cmd",
            "dftk print data.parquet",
            "any input as TSV (or \\texttt{--format})",
            "sh:dftk print data.tsv -o data.parquet && dftk print data.parquet",
        ),
        ("cmd", "dftk transpose", "rows become columns", "sh:dftk transpose data.tsv"),
    ],
)

COLUMNS = (
    "New columns",
    "c3",
    [
        ("sub", "eval: formulas"),
        ("cmd", 'dftk eval -f "rate = tip / total_bill"', "pandas expression", "tips"),
        (
            "cmd",
            'dftk eval -f "m = mean(sepal*)"',
            "row sum, mean, std, min, max, median over columns (globs)",
            "iris",
        ),
        (
            "cmd",
            "dftk eval -f \"big = total_bill > 30\" -f \"size_txt = where(big, 'large', 'small')\"",
            "conditional value",
            "tips",
        ),
        (
            "cmd",
            'dftk eval -f "tip = tip.fillna(0)"',
            "fill gaps; \\texttt{.ffill()}, \\texttt{.interpolate()}",
            "tips",
        ),
        (
            "cmd",
            'dftk eval -f "d = tip.groupby(day).diff()"',
            "change from the row above, per group",
            "tips",
        ),
        (
            "cmd",
            'dftk eval -c "batch = 3" -c \'code = "007"\'',
            "constants (quoted: text)",
            "tips",
        ),
        (
            "cmd",
            'dftk eval -s "n = len(day)"',
            "string, path, DNA and hash functions",
            "tips",
        ),
        (
            "text",
            "Also: \\texttt{chi2\\_to\\_neglogp} \\texttt{fisher\\_test} \\texttt{t2pval} \\ldots\\ (\\texttt{dftk eval -h})",
        ),
        ("sub", "Other column builders"),
        (
            "cmd",
            "dftk func -c tip -g day -t mean",
            "group mean on every row; sum count rank pct\\_rank cumsum \\texttt{qcut:4}",
            "tips",
        ),
        (
            "cmd",
            "dftk binx -c total_bill -b 0:60:10",
            "bin index (or \\texttt{--usevalue m}: bin middle)",
            "tips",
        ),
        (
            "cmd",
            "dftk scale -c tip -g day",
            'z-score; \\texttt{--rank} normal scores; \\texttt{--resid -f "tip \\textasciitilde{} total\\_bill"}',
            "tips",
        ),
        ("cmd", "dftk segid -c day", "id that steps up when the value changes", "tips"),
        (
            "cmd",
            "dftk interp samples.tsv --ref curve.tsv -x x -v y",
            "y at x from a reference curve",
            "sh:dftk interp samples.tsv --ref curve.tsv -x x -v y",
        ),
        (
            "cmd",
            "dftk randvar --dist norm -d noise",
            "random variates (\\texttt{-n 100}: no input)",
            "tips",
        ),
    ],
)


def _table(x0: float, widths, rows, colour: str) -> str:
    """TikZ for a small table at x0 (mm), header row shaded."""
    out = []
    for r, row in enumerate(rows):
        x = x0
        for w, cell in zip(widths, row, strict=True):
            fill = f"{colour}!28" if r == 0 else "white"
            out.append(
                f"\\filldraw[fill={fill},draw={colour}!70] ({x}mm,{-r * 2.7}mm) "
                f"rectangle +({w}mm,-2.7mm) node[midway] {{{cell}}};"
            )
            x += w
    return "\n".join(out)


MELT_PIVOT = (
    "\\par\\smallskip\\begin{center}\\begin{tikzpicture}"
    "[scale=1.2,every node/.style={transform shape},"
    "font=\\fontsize{5.6}{6}\\selectfont\\ttfamily]\n"
    + _table(
        0,
        [4.5, 4.5, 4.5],
        [["id", "a", "b"], ["1", "10", "20"], ["2", "30", "40"]],
        "c4",
    )
    + "\n"
    + _table(
        36,
        [4, 10, 7],
        [
            ["id", "variable", "value"],
            ["1", "a", "10"],
            ["2", "a", "30"],
            ["1", "b", "20"],
            ["2", "b", "40"],
        ],
        "c4",
    )
    + "\n\\draw[-latex,c4,thick] (16mm,-2.4mm) -- node[above] {\\bfseries melt} (34mm,-2.4mm);"
    "\n\\draw[latex-,c4,thick] (16mm,-6.2mm) -- node[below] {\\bfseries pivot} (34mm,-6.2mm);"
    "\n\\end{tikzpicture}\\end{center}\\vspace{-1mm}\n"
)

DATA = (
    "Example data: dftk dataset NAME",
    "c1",
    [
        (
            "text",
            "\\textbf{tips} (244 rows) \\texttt{total\\_bill tip sex smoker day time size}",
        ),
        (
            "text",
            "\\textbf{iris} (150) \\texttt{sepal\\_length sepal\\_width petal\\_length petal\\_width species}",
        ),
        (
            "text",
            "\\textbf{penguins} (344) \\texttt{species island bill\\_length\\_mm bill\\_depth\\_mm flipper\\_length\\_mm body\\_mass\\_g sex}",
        ),
        (
            "text",
            "\\textbf{diamonds} (53\\,940) \\texttt{carat cut color clarity depth table price x y z}",
        ),
        (
            "text",
            "\\textbf{gwas} (120\\,899, synthetic) \\texttt{SNP Chr pos Pval Chisq r2}",
        ),
    ],
)

SQL = (
    "SQL with DuckDB",
    "c8",
    [
        (
            "text",
            '\\texttt{dftk query --sql "..."}: the input is the table \\texttt{data}; a file name in quotes is a table too.',
        ),
        (
            "cmd",
            "SELECT * EXCLUDE (sex, smoker) FROM data",
            "all columns but some",
            "sql:tips",
        ),
        (
            "cmd",
            "SELECT day, count(*) AS n, avg(tip) AS tip FROM data GROUP BY ALL ORDER BY ALL",
            "summary per group",
            "sql:tips",
        ),
        (
            "cmd",
            "SELECT * FROM data QUALIFY row_number() OVER (PARTITION BY day ORDER BY tip DESC) <= 2",
            "top 2 rows per group",
            "sql:tips",
        ),
        (
            "cmd",
            "SELECT *, tip - lag(tip) OVER (PARTITION BY day ORDER BY total_bill) AS dtip FROM data",
            "compare with the row above, per group",
            "sql:tips",
        ),
        (
            "cmd",
            "SELECT * FROM data JOIN 'labels.tsv' USING (day)",
            "join a file",
            "sql:tips",
        ),
        ("cmd", "SELECT * FROM data USING SAMPLE 10", "random rows", "sql:tips"),
    ],
)

RESHAPE = (
    "Combine and reshape",
    "c4",
    [
        (
            "cmd",
            "dftk merge left.tsv -r right.tsv -k id -t left",
            "join on key columns; inner left right outer cross",
            "sh:dftk merge left.tsv -r right.tsv -k id -t left",
        ),
        (
            "cmd",
            "dftk merge left.tsv -r right.tsv -k id --only left",
            "rows only in left (anti-join)",
            "sh:dftk merge left.tsv -r right.tsv -k id --only left",
        ),
        (
            "cmd",
            "dftk concat a.tsv b.tsv --sourcecol src",
            "stack tables; columns matched by name",
            "sh:dftk concat a.tsv b.tsv --sourcecol src",
        ),
        ("latex", MELT_PIVOT),
        ("cmd", "dftk melt -i species", "wide $\\to$ long: variable, value", "iris"),
        (
            "cmd",
            "dftk pivot -i day -g time -v tip -f mean",
            "long $\\to$ wide, with aggregation (columns \\texttt{mean\\_tip\\_Lunch} \\ldots); \\texttt{-z} fills missing cells",
            "tips",
        ),
        (
            "cmd",
            "dftk explode -c go --sep ,",
            "\\texttt{GO:1,GO:2} $\\to$ one row each; \\texttt{implode} joins them back",
            "sh:printf 'gene\\tgo\\nA\\tGO:1,GO:2\\n' | dftk explode -c go --sep ,",
        ),
        (
            "cmd",
            "dftk append -c sample=S1 -c reads=48213",
            "one row; \\texttt{--format json} makes a JSON record",
            "sh:dftk append -c sample=S1 -c reads=48213",
        ),
    ],
)

STATS = (
    "Statistics",
    "c5",
    [
        (
            "cmd",
            "dftk stat -c tip -g day --bootstrap 1000 --randomseed 1",
            "bootstrap CI of the summary",
            "tips",
        ),
        (
            "cmd",
            "dftk wstat -c total_bill -w size -g day",
            "weighted mean, std, quantile CI",
            "tips",
        ),
        (
            "cmd",
            "dftk test -c tip:total_bill -d p --test mannwhitneyu",
            "p-value per pair: student\\_t paired\\_student\\_t wilcoxon kruskal anova kolmogorov\\_smirnov bootstrap \\ldots",
            "tips",
        ),
        (
            "cmd",
            'dftk fit -f "tip ~ total_bill + C(day)"',
            "OLS (R formula); \\texttt{-r} robust, \\texttt{-w} WLS, \\texttt{--anova}, \\texttt{-g}",
            "tips",
        ),
        (
            "cmd",
            "dftk corr -c tip:total_bill --method spearman --ci",
            "correlation, p, CI; \\texttt{--bootstrap 1000}",
            "tips",
        ),
        (
            "cmd",
            "dftk corr -c '*:*'",
            "every pair once; \\texttt{'sepal*:petal*'} cross product",
            "iris",
        ),
        (
            "cmd",
            "dftk corr --all",
            "square matrix for a correlogram (see Plots)",
            "iris",
        ),
        (
            "cmd",
            "dftk corr -c carat:price --tail",
            "tail dependence; plot it with \\texttt{chi}",
            "diamonds",
        ),
        (
            "cmd",
            "dftk pivot -v tip -i day -f mean --bootstrap 1000 --randomseed 1",
            "bootstrap CI of grouped means",
            "tips",
        ),
        (
            "text",
            "p-values are not adjusted for multiple testing; \\texttt{corr} drops missing values pair by pair.",
        ),
    ],
)

PLOTS = (
    "Plots",
    "c6",
    [
        ("img", ["scat.png", "hist.png"]),
        (
            "cmd",
            "dftk scat -x total_bill -y tip -g time --fit",
            "scatter, OLS line per group; \\texttt{--fit lowess}; \\texttt{--hexbin} for big tables",
            "tips",
        ),
        (
            "cmd",
            "dftk hist -x tip -g time -k",
            "histogram, \\texttt{-k} KDE; \\texttt{-b 20}, \\texttt{-n} density, \\texttt{-c} cumulative",
            "tips",
        ),
        (
            "cmd",
            "dftk line -x day -y mean --yerr sem",
            "error bars; \\texttt{--yci lo,hi}",
            "sh:dftk dataset tips -o | dftk stat -c tip -g day -o | dftk line -x day -y mean --yerr sem",
        ),
        ("img", ["line.png", "hexbin.png"]),
        (
            "cmd",
            "dftk scat -x carat -y price --hexbin --hexlog -ly",
            "hexagons",
            "diamonds",
        ),
        ("img", ["heatmap.png", "pairs.png"]),
        (
            "cmd",
            "dftk heatmap -x col1 -y col2 -v correlation --annot --vmin -1 --vmax 1",
            "coloured grid; after \\texttt{corr --all -o}: a correlogram",
            "sh:dftk dataset iris -o | dftk corr --all -o | dftk heatmap -x col1 -y col2 -v correlation --annot --vmin -1 --vmax 1",
        ),
        (
            "cmd",
            "dftk pairs -g species --upper corr",
            "scatter-plot matrix",
            "iris",
        ),
        (
            "cmd",
            "dftk chi -x carat -y price",
            "lower vs upper tail dependence",
            "diamonds",
        ),
        ("img", ["manhattan.png"]),
        (
            "cmd",
            "dftk manhattan",
            "-log10(p) along the genome; \\texttt{qq}: QQ plot, $\\lambda$",
            "gwas",
        ),
        (
            "text",
            "\\textbf{Figure options:} \\texttt{-f fig.png} (or .pdf .svg; no \\texttt{-f}: a window) \\textbullet\\ \\texttt{--size double} or \\texttt{6x4} \\textbullet\\ \\texttt{-t TITLE} \\textbullet\\ \\texttt{-xl -yl} labels \\textbullet\\ \\texttt{-lx -ly} log axes \\textbullet\\ \\texttt{--subgraphcol COL} panels \\textbullet\\ \\texttt{--fontsize presentation}",
        ),
    ],
)

RECIPES = (
    "Pipelines",
    "c7",
    [
        ("sub", "Summarise, then plot"),
        (
            "block",
            [
                "dftk dataset tips -o \\",
                "  | dftk stat -c tip -g day -o \\",
                "  | dftk line -x day -y mean --yerr sem -f tip.png",
            ],
            "sh",
        ),
        ("sub", "Correlogram"),
        (
            "block",
            [
                "dftk dataset iris -o | dftk corr --all -o \\",
                "  | dftk heatmap -x col1 -y col2 -v correlation \\",
                "      --annot --vmin -1 --vmax 1 -f corr.png",
            ],
            "sh",
        ),
        ("sub", "SQL, then a table for a paper"),
        (
            "block",
            [
                "dftk dataset penguins -o | dftk query --sql \\",
                '  "SELECT species, sex, avg(body_mass_g) AS mass',
                "   FROM data WHERE sex IS NOT NULL",
                '   GROUP BY ALL ORDER BY ALL" --digits 4 -o mass.tex',
            ],
            "sh",
        ),
        ("sub", "Status log for a workflow"),
        (
            "block",
            [
                "dftk append runs.tsv --inplace -c step=align \\",
                "  -c status=ok --timestamp",
            ],
            "sh",
        ),
        (
            "text",
            "Creates the file if missing; safe for parallel jobs; \\texttt{.jsonl} and \\texttt{.parquet} too; \\texttt{--strict} refuses new columns.",
        ),
        ("sub", "In a Snakemake rule"),
        (
            "block",
            [
                "rule summary:",
                '    input: "data/{s}.tsv"',
                '    output: "results/{s}.stats.tsv"',
                '    shell: "dftk stat {input} -c x -g grp -o {output}"',
            ],
            None,
        ),
        (
            "text",
            "More patterns: \\texttt{references/snakemake.md} in \\texttt{dftk skill path}.",
        ),
        ("sub", "Tips"),
        (
            "text",
            "\\begin{itemize}"
            "\\item Quote globs and expressions: \\texttt{'*:*'}, \\texttt{\"x > 0\"}."
            "\\item Give FILE first: \\texttt{dftk print f.tsv --sortasc}."
            "\\item \\texttt{set -o pipefail} in scripts (Snakemake does)."
            "\\item A temp pipe file is deleted when read; \\texttt{-o name.parquet} keeps one."
            "\\item \\texttt{dftk annotate f.parquet --set genome=hg38} stores notes in parquet."
            "\\end{itemize}",
        ),
    ],
)

PAGES = [
    [[BASICS, READ], [WRITE, INSPECT], [COLUMNS, SQL]],
    [[RESHAPE, STATS], [PLOTS], [RECIPES, DATA]],
]

# Section colours (title bar and code)
COLOURS = {
    "c1": "2B5D8C",
    "c2": "3C8D5A",
    "c3": "C1631F",
    "c4": "7A4E9B",
    "c5": "B03A48",
    "c6": "1F7F86",
    "c7": "5A5A5A",
    "c8": "8A6D1C",
}

PLOT_FIGURES = {
    "line.png": "dftk dataset tips -o | dftk stat -c tip -g day -o | dftk line -x day -y mean --yerr sem -m o --size 2.6x2.0 -t 'stat | line --yerr'",
    "hexbin.png": "dftk dataset diamonds -o | dftk scat -x carat -y price --hexbin --hexlog -ly --size 2.6x2.0 -t 'scat --hexbin'",
    "scat.png": "dftk dataset tips -o | dftk scat -x total_bill -y tip -g time --fit --size 2.6x2.0 -t 'scat --fit'",
    "hist.png": "dftk dataset tips -o | dftk hist -x tip -g time -k --size 2.6x2.0 -t 'hist -k'",
    "heatmap.png": "dftk dataset iris -o | dftk corr --all -o | dftk heatmap -x col1 -y col2 -v correlation --annot --fmt .1f --vmin -1 --vmax 1 --xlabel '' --ylabel '' --size 3.4x2.9 -t 'heatmap (corr --all)'",
    "manhattan.png": "dftk dataset gwas -o | dftk manhattan --ylim 0 40 --xlabel '' --size 5.4x1.15 -t 'manhattan --ylim 0 40'",
    "pairs.png": "dftk dataset iris -o | dftk pairs -c petal_length petal_width sepal_width -g species --size 3.8x3.1 -t 'pairs -g species'",
}


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

_TT = {
    "\\": "\\textbackslash{}",
    "{": "\\{",
    "}": "\\}",
    "$": "\\$",
    "&": "\\&",
    "#": "\\#",
    "^": "\\textasciicircum{}",
    "_": "\\_",
    "%": "\\%",
    "~": "\\textasciitilde{}",
    "'": "\\textquotesingle{}",
    '"': "\\textquotedbl{}",
    "`": "\\textasciigrave{}",
    "-": "-{}",  # no -- ligature
    " ": "\\ ",
}


def tt(code: str) -> str:
    """Code as LaTeX text: escaped, spaces kept, breaks only at spaces."""
    out = []
    for i, ch in enumerate(code):
        if (
            ch == " "
            and (i == 0 or code[i - 1] != " ")
            and i + 1 < len(code)
            and code[i + 1] != " "
        ):
            out.append(" ")  # a single space may break
        else:
            out.append(_TT.get(ch, ch))
    return "".join(out)


def prose(latex: str) -> str:
    """Description LaTeX: -- stays two hyphens (option names), not a dash."""
    return latex.replace("--", "-{}-")


def code_line(line: str) -> str:
    """A pipeline line: leading spaces kept (glue at a line start is dropped)."""
    stripped = line.lstrip(" ")
    indent = len(line) - len(stripped)
    pad = f"\\phantom{{{'x' * indent}}}" if indent else ""
    return pad + tt(stripped)


def render_item(item, colour: str) -> str:
    kind = item[0]
    if kind == "cmd":
        _, code, desc, _ = item
        return f"\\entry{{{colour}}}{{{tt(code)}}}{{{prose(desc)}}}\n"
    if kind == "latex":
        return item[1]
    if kind == "sub":
        return f"\\subhead{{{colour}}}{{{item[1]}}}\n"
    if kind == "text":
        return f"\\note{{{prose(item[1])}}}\n"
    if kind == "img":
        width = 0.49 if len(item[1]) == 2 else 0.98
        imgs = "\\hfill".join(
            f"\\includegraphics[width={width}\\linewidth]{{{name}}}" for name in item[1]
        )
        return f"\\par\\smallskip\\noindent{imgs}\\par\\smallskip\n"
    if kind == "block":
        lines = "\\\\\n".join(code_line(line) for line in item[1])
        return f"\\codeblock{{{colour}}}{{{lines}}}\n"
    raise ValueError(kind)


def render_section(section) -> str:
    title, colour, items = section
    body = "".join(render_item(item, colour) for item in items)
    return (
        f"\\begin{{sheetsection}}{{{colour}}}{{{title}}}\n{body}\\end{{sheetsection}}\n"
    )


def render(version: str) -> str:
    colours = "".join(
        f"\\definecolor{{{name}}}{{HTML}}{{{value}}}\n"
        for name, value in COLOURS.items()
    )
    pages = []
    for number, columns in enumerate(PAGES):
        cols = "\\hfill\n".join(
            "\\begin{minipage}[t]{0.322\\linewidth}\\vspace{0pt}\n"
            + "".join(render_section(s) for s in sections)
            + "\\end{minipage}\n"
            for sections in columns
        )
        header = "\\header\n" if number == 0 else ""
        pages.append(f"{header}\\noindent\n{cols}")
    body = "\\newpage\n".join(pages)
    return (
        (HERE / "preamble.tex")
        .read_text()
        .replace("%COLOURS%", colours)
        .replace("%VERSION%", version)
        + "\\begin{document}\n"
        + body
        + "\\end{document}\n"
    )


# ---------------------------------------------------------------------------
# Checks, figures, build
# ---------------------------------------------------------------------------


def _commands():
    """(label, shell command) for every example."""
    for page in PAGES:
        for column in page:
            for title, _, items in column:
                for item in items:
                    if item[0] == "cmd" and item[3]:
                        code, check = item[1], item[3]
                        if check.startswith("sh:"):
                            yield code, check[3:]
                        elif check.startswith("sql:"):
                            yield (
                                code,
                                f'dftk dataset {check[4:]} -o | dftk query --sql "{code}"',
                            )
                        elif code.startswith("dftk "):
                            yield code, f"dftk dataset {check} -o | {code}"
                        else:  # an option of every command
                            yield code, f"dftk dataset {check} -o | dftk print {code}"
                    elif item[0] == "block" and item[2]:
                        yield title, "\n".join(item[1])


def check_examples() -> None:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in FILES.items():
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(os.path.join(tmp, name), mode) as fh:
                fh.write(content)
        for label, command in _commands():
            result = subprocess.run(
                ["bash", "-o", "pipefail", "-c", command],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
            )
            ok = result.returncode == 0 and "ERROR" not in result.stderr
            if not ok:
                failed += 1
                print(
                    f"FAILED: {label}\n  $ {command}\n  {result.stderr.strip()[-400:]}"
                )
    if failed:
        sys.exit(f"{failed} example(s) failed")
    print("all examples ran")


def draw_figures() -> None:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    for name, command in PLOT_FIGURES.items():
        subprocess.run(
            [
                "bash",
                "-o",
                "pipefail",
                "-c",
                f"{command} --fontsize publication -f {name}",
            ],
            cwd=HERE,
            env=env,
            check=True,
            capture_output=True,
        )


def build() -> None:
    version = subprocess.run(
        ["dftk", "--version"], capture_output=True, text=True, check=True
    ).stdout.split()[-1]
    (HERE / "cheatsheet.tex").write_text(render(version))
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "cheatsheet.tex"],
        cwd=HERE,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode:
        sys.exit(result.stdout[-3000:])
    os.replace(HERE / "cheatsheet.pdf", HERE / "dftk-cheatsheet.pdf")
    for ext in ("aux", "log"):
        (HERE / f"cheatsheet.{ext}").unlink(missing_ok=True)
    pages = result.stdout.count("[") and result.stdout.split("Output written on")[-1]
    print("built dftk-cheatsheet.pdf:", pages.strip().splitlines()[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--no-check", action="store_true", help="Don't run the examples"
    )
    parser.add_argument("--no-figures", action="store_true", help="Keep the thumbnails")
    opts = parser.parse_args()
    if not opts.no_check:
        check_examples()
    if not opts.no_figures:
        draw_figures()
    build()

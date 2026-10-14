# dftk in Snakemake and Make

## Principles

- **One rule per result someone else depends on.** Write it to a named file
  (`-o work/clean.parquet`) so Snakemake can track it, rerun only what changed
  and run independent rules in parallel.
- **Pipe inside a rule (`-o | ...`) only for steps nothing else needs.** A
  short chain (filter, summarise, add a label) in one rule avoids writing
  files nobody reads. Split it into separate rules as soon as an intermediate
  table is reused, is expensive to recompute, or is worth inspecting.
- **Parquet between rules, TSV for final tables.** Parquet keeps exact
  column types and carries provenance metadata; TSV is easy to read and diff.
  `-o x.parquet` writes parquet, `-o x.tsv` writes TSV.
- **`-o FILE.parquet` prints the file's path to stdout.** That is harmless,
  but `> /dev/null` keeps the Snakemake log clean. Send stderr to `{log}`.
- **Record provenance with `--meta`.** `--meta rule={rule}` (Snakemake
  fills in the rule name) or `--meta build=hg38`; metadata from earlier steps
  is carried forward automatically. Read it back with
  `dftk annotate FILE.parquet`.
- **Failures stop the rule.** Snakemake runs shell commands with
  `set -euo pipefail`, so a failing step anywhere in a pipe fails the rule and
  Snakemake removes its partial outputs. The first `Input error:` line in the
  log is the cause.
- **Plots always need `-f {output}`**; without it nothing is written.
- **Temp pipe files** go to `$DFTK_TMPDIR` (default `$TMPDIR/dftk`). On a
  cluster, every command of one pipe runs in the same job, so the default is
  fine. Set `DFTK_TMPDIR` only if `$TMPDIR` is small or shared.
- **Quoting:** wrap dftk expressions in double quotes inside the rule's
  `shell:` string and use single quotes for string literals in the
  expression: `-q "species == '{wildcards.species}'"`. A literal brace in a
  Snakemake `shell:` string must be doubled (`{{` `}}`).

## Example Snakefile

Runs as-is (it starts from a bundled dataset). In a real project the first
rule would be your own data file, read directly by the next rule.

```python
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]


rule all:
    input:
        "results/summary.tsv",
        "results/fit.tsv",
        "results/plots/bill.png",


# Raw input. In a real project this would be your own file, read directly.
rule load:
    output:
        "data/penguins.parquet",
    log:
        "logs/load.log",
    shell:
        "dftk dataset penguins -o {output} --meta source=penguins > /dev/null 2> {log}"


rule clean:
    input:
        "data/penguins.parquet",
    output:
        "work/clean.parquet",
    log:
        "logs/clean.log",
    shell:
        """
        dftk query {input} -q "sex.notna()" \
            -o {output} --meta rule={rule} > /dev/null 2> {log}
        """


# One job per wildcard value; a short pipe inside one rule needs no
# intermediate file because nothing else depends on it.
rule species_stats:
    input:
        "work/clean.parquet",
    output:
        "work/stats/{species}.tsv",
    log:
        "logs/species_stats/{species}.log",
    shell:
        """
        (dftk query {input} -q "species == '{wildcards.species}'" -o \
            | dftk stat ... -c bill_length_mm body_mass_g -g sex -o \
            | dftk eval ... -c "species = {wildcards.species}" \
                --select species + -o {output}) 2> {log}
        """


rule summary:
    input:
        expand("work/stats/{species}.tsv", species=SPECIES),
    output:
        "results/summary.tsv",
    log:
        "logs/summary.log",
    shell:
        "dftk concat {input} --sortasc name species sex -o {output} 2> {log}"


rule fit:
    input:
        "work/clean.parquet",
    output:
        "results/fit.tsv",
    log:
        "logs/fit.log",
    shell:
        """
        dftk fit {input} -f "body_mass_g ~ flipper_length_mm" -g species \
            -o {output} 2> {log}
        """


rule plot:
    input:
        "work/clean.parquet",
    output:
        "results/plots/bill.png",
    log:
        "logs/plot.log",
    shell:
        """
        dftk scat {input} -x bill_length_mm -y bill_depth_mm -g species \
            -f {output} 2> {log}
        """
```

Notes on the example:

- `species_stats` wraps the pipe in `( … ) 2> {log}` so stderr from every
  step reaches the log, not just the last one.
- The per-species split could also be done in a single rule with `-g species`
  (`dftk stat work/clean.parquet -c … -g species sex`). Use wildcards when the
  per-group work is heavy, should run in parallel, or each group gets its own
  downstream rules.
- `dftk split` writes one file per group, but Snakemake cannot know those
  output names in advance. Prefer wildcards plus `-q`/`--prequery` filtering,
  or declare the split outputs explicitly when the groups are fixed.

## Make

The same rules in a Makefile. Set bash with `pipefail` so a failing step
inside a pipe fails the target, and `.DELETE_ON_ERROR` so a failed command
does not leave a partial output that looks up to date.

```make
SHELL := bash
.SHELLFLAGS := -euo pipefail -c
.DELETE_ON_ERROR:

all: results/summary.tsv

data/penguins.parquet:
	mkdir -p $(@D)
	dftk dataset penguins -o $@ --meta source=penguins > /dev/null

work/clean.parquet: data/penguins.parquet
	mkdir -p $(@D)
	dftk query $< -q "sex.notna()" -o $@ --meta step=clean > /dev/null

results/summary.tsv: work/clean.parquet
	mkdir -p $(@D)
	dftk stat $< -c bill_length_mm body_mass_g -g species sex -o $@
```

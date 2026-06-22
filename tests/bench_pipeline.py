#!/usr/bin/env python3
"""
bench_pipeline.py — wall-clock benchmark comparing TSV stdin/stdout vs
parquet intermediate files across multi-step dfstat pipelines.

Usage
-----
    cd stattools
    python tests/bench_pipeline.py
    python tests/bench_pipeline.py --rows 2000000
    python tests/bench_pipeline.py --rows 500000 --runs 3
    python tests/bench_pipeline.py --pipeline A        # run only pipeline A
    python tests/bench_pipeline.py --validate          # check TSV==parquet output shape

Transport modes
---------------
  tsv    Commands connected with |, each stage writing TSV to stdout.
  parq   Commands connected with |, each intermediate stage using -o to
         write a temp parquet file and print its path to stdout.

The two modes produce identical DataFrames; only the inter-stage transport
format changes.  Parquet is expected to win on large data because binary
columnar encoding avoids the text-serialisation/parsing overhead of TSV.
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Pipeline definitions
# ---------------------------------------------------------------------------


@dataclass
class Pipeline:
    name: str
    description: str
    # Shell command fragments.  Use {N} for row count.
    stages: list[str]
    # Row count for this pipeline (overrides --rows when set).
    rows: int | None = None


PIPELINES: list[Pipeline] = [
    Pipeline(
        name="A",
        description=(
            "Wide generation → eval (2 derived cols) → melt to long format.\n"
            "  randvar(x1,x2,x3,g) | eval(prod,norm_x3) | melt(-i g)\n"
            "  Inter-stage TSV has many columns × many rows — parquet wins big."
        ),
        stages=[
            "dfstat randvar -n {N} -d x1 --dist norm --randomseed 11",
            "dfstat randvar - -d x2 --dist norm --parameters loc:2,scale:0.5 --randomseed 22",  # noqa: E501
            "dfstat randvar - -d x3 --dist uniform --parameters loc:-1,scale:2 --randomseed 33",  # noqa: E501
            "dfstat randvar - -d g --dist randint --parameters low:0,high:4 --randomseed 44",  # noqa: E501
            "dfstat eval - -f 'prod = x1 * x2' -f 'norm_x3 = x3 / (abs(x3) + 1.0)'",
            "dfstat melt - -i g -d feature -v value",
        ],
        rows=1_000_000,
    ),
    Pipeline(
        name="B",
        description=(
            "Generation → func (qcut:10 decile bins) → pivot (count by decile×group).\n"
            "  randvar(x,grp) | func(x→decile) | pivot(-v x -i decile -g grp -f count)\n"  # noqa: E501
            "  Large ingest; tiny pivot output — tests read cost."
        ),
        stages=[
            "dfstat randvar -n {N} -d x --dist norm --randomseed 55",
            "dfstat randvar - -d grp --dist randint --parameters low:0,high:5 --randomseed 66",  # noqa: E501
            "dfstat func - -c x -t qcut:10 -d decile",
            "dfstat pivot - -v x -i decile -g grp -f count",
        ],
        rows=2_000_000,
    ),
    Pipeline(
        name="C",
        description=(
            "Wide data → eval (diff + ratio) → stat (mean/std/CI by group).\n"
            "  randvar(a,b,g) | eval(diff,ratio) | stat(-c diff ratio -g g)\n"
            "  Three large inter-stage transfers, tiny output."
        ),
        stages=[
            "dfstat randvar -n {N} -d a --dist norm --parameters loc:0,scale:1 --randomseed 77",  # noqa: E501
            "dfstat randvar - -d b --dist norm --parameters loc:5,scale:2 --randomseed 88",  # noqa: E501
            "dfstat randvar - -d g --dist randint --parameters low:0,high:3 --randomseed 99",  # noqa: E501
            "dfstat eval - -f 'diff = a - b' -f 'ratio = a / (abs(b) + 0.001)'",
            "dfstat stat - -c diff ratio -g g",
        ],
        rows=3_000_000,
    ),
    Pipeline(
        name="D",
        description=(
            "Gigabyte-scale stress test: 10 M rows × 8 float columns.\n"
            "  randvar×8 cols | eval(3 cols) | pivot(mean by group×quartile)\n"
            "  Designed to push inter-stage transfer toward ~1 GB TSV."
        ),
        stages=[
            "dfstat randvar -n {N} -d v1 --dist norm --randomseed 101",
            "dfstat randvar - -d v2 --dist norm --parameters loc:1,scale:1 --randomseed 102",  # noqa: E501
            "dfstat randvar - -d v3 --dist norm --parameters loc:-1,scale:2 --randomseed 103",  # noqa: E501
            "dfstat randvar - -d v4 --dist uniform --parameters loc:0,scale:10 --randomseed 104",  # noqa: E501
            "dfstat randvar - -d v5 --dist norm --randomseed 105",
            "dfstat randvar - -d g --dist randint --parameters low:0,high:5 --randomseed 106",  # noqa: E501
            "dfstat randvar - -d q --dist randint --parameters low:1,high:5 --randomseed 107",  # noqa: E501
            "dfstat eval - -f 'sum5 = v1 + v2 + v3 + v4 + v5'"
            " -f 'absratio = abs(v1) / (abs(v2) + 0.001)'"
            " -f 'interact = v3 * v4'",
            "dfstat pivot - -v sum5 absratio interact -i g -g q -f mean",
        ],
        rows=5_000_000,
    ),
]

PIPELINE_MAP = {p.name: p for p in PIPELINES}


# ---------------------------------------------------------------------------
# Shell-pipeline builder
# ---------------------------------------------------------------------------


def _build_cmd(stages: list[str], parquet: bool, rows: int) -> str:
    """Return a shell command string joining *stages* with pipe.

    In parquet mode every stage except the last has -o appended so it writes a
    temp parquet and prints the path to stdout for the next stage to consume.
    """
    expanded = [s.format(N=rows) for s in stages]
    if parquet:
        piped = [s + " -o" for s in expanded[:-1]] + [expanded[-1]]
    else:
        piped = expanded
    return " | ".join(piped)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    pipeline: str
    mode: str  # "tsv" or "parq"
    rows: int
    elapsed: float  # seconds
    returncode: int
    stderr_tail: str  # last few lines of stderr for diagnostics


def run_pipeline(
    pipeline: Pipeline,
    rows: int,
    parquet: bool,
    capture_output: bool = True,
    quiet: bool = False,
) -> RunResult:
    """Run *pipeline* in the given transport mode and return timing info."""
    mode = "parq" if parquet else "tsv "
    cmd = _build_cmd(pipeline.stages, parquet=parquet, rows=rows)

    if not quiet:
        label = f"[{pipeline.name}] {mode}"
        print(f"  {label}  running …", end="", flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    stderr_tail = ""
    if result.stderr:
        lines = result.stderr.strip().splitlines()
        stderr_tail = "\n".join(lines[-5:])

    if not quiet:
        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"\r  {label}  {elapsed:6.2f}s  {status}")

    return RunResult(
        pipeline=pipeline.name,
        mode=mode.strip(),
        rows=rows,
        elapsed=elapsed,
        returncode=result.returncode,
        stderr_tail=stderr_tail,
    )


# ---------------------------------------------------------------------------
# Validation (shape check: TSV and parquet must produce same row×col count)
# ---------------------------------------------------------------------------


def validate_pipeline(pipeline: Pipeline, rows: int) -> bool:
    """Run both modes and compare output shape."""

    def _run_capture(parquet: bool) -> str:
        cmd = _build_cmd(pipeline.stages, parquet=parquet, rows=rows)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return r.stdout, r.returncode

    print(f"  Validating pipeline {pipeline.name} …", end="", flush=True)
    tsv_out, rc_tsv = _run_capture(parquet=False)
    parq_out, rc_parq = _run_capture(parquet=True)

    if rc_tsv != 0 or rc_parq != 0:
        print(f"  FAILED (rc tsv={rc_tsv} parq={rc_parq})")
        return False

    tsv_lines = [line for line in tsv_out.strip().splitlines() if line]
    parq_lines = [line for line in parq_out.strip().splitlines() if line]

    if len(tsv_lines) != len(parq_lines):
        print(f"\n  MISMATCH: tsv={len(tsv_lines)} lines, parq={len(parq_lines)} lines")
        return False

    if tsv_lines and tsv_lines[0] != parq_lines[0]:
        print(
            f"\n  HEADER MISMATCH:\n    tsv : {tsv_lines[0]}\n    parq: {parq_lines[0]}"
        )
        return False

    hdr = tsv_lines[0] if tsv_lines else "(empty)"
    print(f"  OK  ({len(tsv_lines) - 1} rows, header: {hdr})")
    return True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[RunResult]) -> None:
    print()
    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)

    header = f"  {'Pipeline':<12} {'Rows':>10}  {'TSV (s)':>10}  {'Parq (s)':>10}  {'Speedup':>9}"  # noqa: E501
    print(header)
    print("  " + "-" * 68)

    # Group by pipeline name
    by_pipeline: dict[str, dict[str, RunResult]] = {}
    for r in results:
        by_pipeline.setdefault(r.pipeline, {})[r.mode] = r

    for pname in sorted(by_pipeline):
        modes = by_pipeline[pname]
        tsv_r = modes.get("tsv")
        parq_r = modes.get("parq")
        rows = (tsv_r or parq_r).rows

        tsv_s = f"{tsv_r.elapsed:10.2f}" if tsv_r else "        —"
        parq_s = f"{parq_r.elapsed:10.2f}" if parq_r else "        —"

        if tsv_r and parq_r and parq_r.elapsed > 0:
            speedup = tsv_r.elapsed / parq_r.elapsed
            speedup_s = f"{speedup:8.2f}×"
        else:
            speedup_s = "       —"

        print(f"  {pname:<12} {rows:>10}  {tsv_s}  {parq_s}  {speedup_s}")

    print()
    failed = [r for r in results if r.returncode != 0]
    if failed:
        print("  FAILURES:")
        for r in failed:
            print(f"    [{r.pipeline}] {r.mode}  rc={r.returncode}")
            if r.stderr_tail:
                for line in r.stderr_tail.splitlines():
                    print(f"      {line}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rows",
        "-n",
        type=int,
        default=None,
        help=(
            "Override number of rows for all pipelines.  "
            "Default: each pipeline's built-in row count."
        ),
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=1,
        help="Number of timed runs per (pipeline, mode) pair (default: 1).",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        nargs="*",
        choices=sorted(PIPELINE_MAP),
        metavar="NAME",
        help="Run only the named pipeline(s) (A/B/C/D).  Default: all.",
    )
    parser.add_argument(
        "--no-tsv",
        action="store_true",
        help="Skip TSV mode.",
    )
    parser.add_argument(
        "--no-parq",
        action="store_true",
        help="Skip parquet mode.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Before timing, verify that both modes produce identical output shape.  "
            "Uses a small row count (10 000) to keep it fast."
        ),
    )
    parser.add_argument(
        "--show-cmds",
        action="store_true",
        help="Print the shell commands that would be run, then exit.",
    )
    args = parser.parse_args(argv)

    selected = [PIPELINE_MAP[n] for n in args.pipeline] if args.pipeline else PIPELINES

    # ---- show commands and exit ------------------------------------------
    if args.show_cmds:
        for p in selected:
            rows = args.rows or p.rows
            print(f"\n{'─' * 70}")
            print(f"Pipeline {p.name}: {p.description.splitlines()[0]}")
            print(f"{'─' * 70}")
            print(f"  TSV  : {_build_cmd(p.stages, parquet=False, rows=rows)}")
            print(f"  PARQ : {_build_cmd(p.stages, parquet=True, rows=rows)}")
        return 0

    # ---- validation pass -------------------------------------------------
    if args.validate:
        print("\nValidation (10 000 rows):")
        ok = True
        for p in selected:
            ok &= validate_pipeline(p, rows=10_000)
        if not ok:
            print("\nValidation FAILED — aborting benchmark.")
            return 1
        print()

    # ---- benchmark -------------------------------------------------------
    all_results: list[RunResult] = []

    for p in selected:
        rows = args.rows or p.rows
        print(f"\nPipeline {p.name}: {p.description.splitlines()[0]}")
        print(f"  rows={rows:,}  runs={args.runs}")

        for run_i in range(args.runs):
            if args.runs > 1:
                print(f"  run {run_i + 1}/{args.runs}:")
            if not args.no_tsv:
                r = run_pipeline(p, rows=rows, parquet=False)
                all_results.append(r)
            if not args.no_parq:
                r = run_pipeline(p, rows=rows, parquet=True)
                all_results.append(r)

    print_report(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

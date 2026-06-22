"""
stattools.commands.annotate_cmd — dfstat annotate subcommand.

Read and write provenance metadata embedded in parquet files.

Parquet stores arbitrary key-value string pairs in the file-level schema
metadata.  dfstat uses this to carry provenance annotations (genome build,
source dataset, processing steps, etc.) through a pipeline.

Usage examples:

  # List all annotations on a file
  dfstat annotate data.parquet

  # Set / update annotations (modifies the file in-place)
  dfstat annotate data.parquet --set genome=hg38 --set source=gwas_2024

  # Get a single annotation
  dfstat annotate data.parquet --get genome

  # Remove an annotation
  dfstat annotate data.parquet --delete genome

  # Embed annotations when writing parquet from any dfstat command
  dfstat eval data.tsv -f "z = x + y" -o result.parquet \\
      --meta genome=hg38 --meta source=my_pipeline

Annotations propagate through the parquet pipe: a file annotated at
creation is re-annotated on every subsequent -o write unless explicitly
overridden.
"""

import argparse
import sys

import pyarrow.parquet as pq

from stattools.commands.base import BaseCommand

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_meta(path: str) -> dict[str, str]:
    """Return custom (non-pandas) schema metadata from *path*."""
    schema = pq.read_schema(path)
    raw = schema.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items() if k != b"pandas"}


def _write_meta(path: str, meta: dict[str, str]) -> None:
    """Overwrite the custom metadata in the parquet file at *path* in-place."""
    table = pq.read_table(path)
    existing = table.schema.metadata or {}
    # Keep the pandas key intact; replace everything else with *meta*
    pandas_meta = {k: v for k, v in existing.items() if k == b"pandas"}
    new_meta = {
        **pandas_meta,
        **{k.encode(): v.encode() for k, v in meta.items()},
    }
    table = table.replace_schema_metadata(new_meta)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class AnnotateCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "annotate"

    @property
    def help(self) -> str:
        return "Read and write provenance metadata in parquet files"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        import argparse as ap

        parser.formatter_class = ap.RawDescriptionHelpFormatter
        parser.description = __doc__

        parser.add_argument(
            "PARQUET",
            help="Parquet file to inspect or annotate.",
        )
        parser.add_argument(
            "--set",
            dest="set_meta",
            help="Set a metadata annotation (KEY=VALUE).  Repeatable.",
            default=[],
            action="append",
            metavar="KEY=VALUE",
        )
        parser.add_argument(
            "--get",
            dest="get_key",
            help="Print the value of a single annotation key.",
            default=None,
            metavar="KEY",
        )
        parser.add_argument(
            "--delete",
            dest="delete_keys",
            help="Remove an annotation key.  Repeatable.",
            default=[],
            action="append",
            metavar="KEY",
        )
        parser.add_argument(
            "--clear",
            help="Remove ALL custom annotations from the file.",
            action="store_true",
        )

    def execute(self, args: argparse.Namespace) -> None:
        path: str = args.PARQUET

        if not path.endswith(".parquet"):
            print(
                f"annotate: {path!r} does not appear to be a parquet file.",
                file=sys.stderr,
            )
            sys.exit(1)

        # ---- mutations (--set / --delete / --clear) — applied first -------

        if args.clear or args.set_meta or args.delete_keys:
            meta = {} if args.clear else _read_meta(path)

            for item in args.set_meta:
                try:
                    k, v = item.split("=", 1)
                    meta[k.strip()] = v.strip()
                except ValueError:
                    print(
                        f"annotate --set: expected KEY=VALUE, got {item!r}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            for key in args.delete_keys:
                meta.pop(key, None)

            _write_meta(path, meta)

        # ---- read current state and respond --------------------------------

        meta = _read_meta(path)

        if args.get_key is not None:
            value = meta.get(args.get_key)
            if value is None:
                print(
                    f"annotate --get: key {args.get_key!r} not found.", file=sys.stderr
                )
                sys.exit(1)
            print(value)
        else:
            # Default: list all annotations as TSV (key sorted)
            for k, v in sorted(meta.items()):
                print(f"{k}\t{v}")

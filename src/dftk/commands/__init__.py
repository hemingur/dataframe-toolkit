# dftk.commands — registry of all dftk subcommands.
#
# Commands are registered by name and import path instead of being imported
# here, so that running one command imports only that command's module and its
# dependencies (scipy, statsmodels, matplotlib, …), not every command's.
#
# To add a new subcommand:
#   1. Create src/dftk/commands/<name>_cmd.py with a class that inherits
#      from BaseCommand (see base.py for the contract and an example skeleton).
#   2. Add a ("<name>", "<name>_cmd:<ClassName>") entry to COMMAND_GROUPS below.
#   That's it — cli.py discovers commands solely through this registry.
#   tests/commands/test_registry.py checks that every entry imports and that
#   every *_cmd.py module is registered.

import importlib

from dftk.commands.base import BaseCommand

# Ported subcommands — add as each one is implemented:
#   ("color", "color_cmd:ColorCommand")
#   ("wavelet", "wavelet_cmd:WaveletCommand")
#   ("fisher", ...)  — superseded by eval's fisher_test/fisher_OR

COMMAND_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Data transformation",
        [
            ("eval", "eval_cmd:EvalCommand"),
            ("query", "query_cmd:QueryCommand"),
            ("merge", "merge_cmd:MergeCommand"),
            ("concat", "concat_cmd:ConcatCommand"),
            ("melt", "melt_cmd:MeltCommand"),
            ("explode", "explode_cmd:ExplodeCommand"),
            ("implode", "implode_cmd:ImplodeCommand"),
            ("pivot", "pivot_cmd:PivotCommand"),
            ("func", "func_cmd:FuncCommand"),
            ("scale", "scale_cmd:ScaleCommand"),
            ("interp", "interp_cmd:InterpCommand"),
            ("binx", "binx_cmd:BinxCommand"),
            ("transpose", "transpose_cmd:TransposeCommand"),
            ("segid", "segid_cmd:SegidCommand"),
            ("cleannames", "cleannames_cmd:CleannamesCommand"),
        ],
    ),
    (
        "Statistics",
        [
            ("stat", "stat_cmd:StatCommand"),
            ("wstat", "wstat_cmd:WstatCommand"),
            ("fit", "fit_cmd:FitCommand"),
            ("test", "test_cmd:TestCommand"),
            ("corr", "corr_cmd:CorrCommand"),
            ("describe", "describe_cmd:DescribeCommand"),
            ("info", "info_cmd:InfoCommand"),
            ("randvar", "randvar_cmd:RandvarCommand"),
        ],
    ),
    (
        "Plots",
        [
            ("scat", "scat_cmd:ScatCommand"),
            ("line", "line_cmd:LineCommand"),
            ("hist", "hist_cmd:HistCommand"),
            ("chi", "chi_cmd:ChiCommand"),
        ],
    ),
    (
        "Utilities",
        [
            ("dataset", "dataset_cmd:DatasetCommand"),
            ("sample", "sample_cmd:SampleCommand"),
            ("split", "split_cmd:SplitCommand"),
            ("annotate", "annotate_cmd:AnnotateCommand"),
            ("print", "print_cmd:PrintCommand"),
            ("clean", "clean_cmd:CleanCommand"),
        ],
    ),
]

# `help` is a command like the others but is not listed under a group heading.
_REGISTRY: dict[str, str] = dict(
    [entry for _, entries in COMMAND_GROUPS for entry in entries]
    + [("help", "help:HelpCommand")]
)

COMMAND_NAMES: list[str] = list(_REGISTRY)


def load_command(name: str) -> BaseCommand:
    """Import and instantiate the command registered as *name*."""
    module_name, class_name = _REGISTRY[name].split(":")
    module = importlib.import_module(f"dftk.commands.{module_name}")
    return getattr(module, class_name)()


def load_command_groups() -> list[tuple[str, list[BaseCommand]]]:
    """Import and instantiate every grouped command (slow: imports everything)."""
    return [
        (header, [load_command(name) for name, _ in entries])
        for header, entries in COMMAND_GROUPS
    ]

# dftk.commands — explicit registry of all dftk subcommands.
#
# To add a new subcommand:
#   1. Create src/dftk/commands/<name>.py with a class that inherits
#      from BaseCommand (see base.py for the contract and an example skeleton).
#   2. Import the class here and add an instance to command_list below.
#   That's it — cli.py discovers commands solely through this COMMANDS list.

from dftk.commands.annotate_cmd import AnnotateCommand
from dftk.commands.binx_cmd import BinxCommand
from dftk.commands.clean_cmd import CleanCommand
from dftk.commands.concat_cmd import ConcatCommand
from dftk.commands.corr_cmd import CorrCommand
from dftk.commands.dataset_cmd import DatasetCommand
from dftk.commands.describe_cmd import DescribeCommand
from dftk.commands.eval_cmd import EvalCommand
from dftk.commands.fit_cmd import FitCommand
from dftk.commands.func_cmd import FuncCommand
from dftk.commands.help import HelpCommand
from dftk.commands.hist_cmd import HistCommand
from dftk.commands.interp_cmd import InterpCommand
from dftk.commands.line_cmd import LineCommand
from dftk.commands.melt_cmd import MeltCommand
from dftk.commands.merge_cmd import MergeCommand
from dftk.commands.pivot_cmd import PivotCommand
from dftk.commands.print_cmd import PrintCommand
from dftk.commands.query_cmd import QueryCommand
from dftk.commands.randvar_cmd import RandvarCommand
from dftk.commands.sample_cmd import SampleCommand
from dftk.commands.scale_cmd import ScaleCommand
from dftk.commands.scat_cmd import ScatCommand
from dftk.commands.split_cmd import SplitCommand
from dftk.commands.stat_cmd import StatCommand
from dftk.commands.test_cmd import TestCommand
from dftk.commands.wstat_cmd import WstatCommand

# Ported subcommands — uncomment as each one is added:
# from dftk.commands.segid     import SegidCommand
# from dftk.commands.info      import InfoCommand
# from dftk.commands.transpose import TransposeCommand
# from dftk.commands.color     import ColorCommand
# from dftk.commands.wavelet   import WaveletCommand
# from dftk.commands.fisher    import FisherCommand  # superseded by eval's
#                                                         # fisher_test/fisher_OR

COMMAND_GROUPS: list[tuple[str, list]] = [
    (
        "Data transformation",
        [
            EvalCommand(),
            QueryCommand(),
            MergeCommand(),
            ConcatCommand(),
            MeltCommand(),
            PivotCommand(),
            FuncCommand(),
            ScaleCommand(),
            InterpCommand(),
            BinxCommand(),
        ],
    ),
    (
        "Statistics",
        [
            StatCommand(),
            WstatCommand(),
            FitCommand(),
            TestCommand(),
            CorrCommand(),
            DescribeCommand(),
            RandvarCommand(),
        ],
    ),
    (
        "Plots",
        [
            ScatCommand(),
            LineCommand(),
            HistCommand(),
        ],
    ),
    (
        "Utilities",
        [
            DatasetCommand(),
            SampleCommand(),
            SplitCommand(),
            AnnotateCommand(),
            PrintCommand(),
            CleanCommand(),
        ],
    ),
]

command_list = [cmd for _, cmds in COMMAND_GROUPS for cmd in cmds]

# HelpCommand receives the live list and groups so it can introspect any peer.
help_command = HelpCommand(command_list, COMMAND_GROUPS)

COMMANDS = command_list + [help_command]

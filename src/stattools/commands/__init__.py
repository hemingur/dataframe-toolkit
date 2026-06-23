# stattools.commands — explicit registry of all dfstat subcommands.
#
# To add a new subcommand:
#   1. Create src/stattools/commands/<name>.py with a class that inherits
#      from BaseCommand (see base.py for the contract and an example skeleton).
#   2. Import the class here and add an instance to command_list below.
#   That's it — cli.py discovers commands solely through this COMMANDS list.

from stattools.commands.annotate_cmd import AnnotateCommand
from stattools.commands.clean_cmd import CleanCommand
from stattools.commands.concat_cmd import ConcatCommand
from stattools.commands.dataset_cmd import DatasetCommand
from stattools.commands.describe_cmd import DescribeCommand
from stattools.commands.eval_cmd import EvalCommand
from stattools.commands.fit_cmd import FitCommand
from stattools.commands.func_cmd import FuncCommand
from stattools.commands.help import HelpCommand
from stattools.commands.hist_cmd import HistCommand
from stattools.commands.interp_cmd import InterpCommand
from stattools.commands.line_cmd import LineCommand
from stattools.commands.melt_cmd import MeltCommand
from stattools.commands.merge_cmd import MergeCommand
from stattools.commands.pivot_cmd import PivotCommand
from stattools.commands.print_cmd import PrintCommand
from stattools.commands.query_cmd import QueryCommand
from stattools.commands.randvar_cmd import RandvarCommand
from stattools.commands.sample_cmd import SampleCommand
from stattools.commands.scale_cmd import ScaleCommand
from stattools.commands.scat_cmd import ScatCommand
from stattools.commands.stat_cmd import StatCommand
from stattools.commands.test_cmd import TestCommand

# Ported subcommands — uncomment as each one is added:
# from stattools.commands.split     import SplitCommand
# from stattools.commands.corr      import CorrCommand
# from stattools.commands.wstat     import WstatCommand
# from stattools.commands.binx      import BinxCommand
# from stattools.commands.segid     import SegidCommand
# from stattools.commands.info      import InfoCommand
# from stattools.commands.transpose import TransposeCommand
# from stattools.commands.color     import ColorCommand
# from stattools.commands.wavelet   import WaveletCommand
# from stattools.commands.fisher    import FisherCommand  # superseded by eval's
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
        ],
    ),
    (
        "Statistics",
        [
            StatCommand(),
            FitCommand(),
            TestCommand(),
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

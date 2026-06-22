# This module has moved to stattools.common.io.
# This file exists only as a compatibility shim and can be deleted once
# nothing imports from stattools.io directly.
from stattools.common.io import (  # noqa: F401
    DFSTAT_TMPDIR,
    _is_pipe_path,
    _write_pipe_parquet,
    globnames,
    io,
)

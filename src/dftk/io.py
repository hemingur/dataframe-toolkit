# This module has moved to dftk.common.io.
# This file exists only as a compatibility shim and can be deleted once
# nothing imports from dftk.io directly.
from dftk.common.io import (  # noqa: F401
    DFTK_TMPDIR,
    _is_pipe_path,
    _write_pipe_parquet,
    globnames,
    io,
)

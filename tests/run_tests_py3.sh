#!/usr/bin/env bash
# This file can be used by developers to run all tests locally before travis

SCRIPT_DIR=$(dirname -- "$(realpath -- "$0")")

#python3 "$SCRIPT_DIR"/parser/test_headers.py
#python3 "$SCRIPT_DIR"/parser/test_openmp.py
#python3 "$SCRIPT_DIR"/parser/test_openacc.py
#python3 "$SCRIPT_DIR"/syntax/test_syntax.py
#python3 "$SCRIPT_DIR"/semantic/test_semantic.py
#python3 "$SCRIPT_DIR"/codegen/test_codegen.py
#python3 "$SCRIPT_DIR"/errors/test_errors.py
#python3 "$SCRIPT_DIR"/warnings/test_warnings.py
#python3 "$SCRIPT_DIR"/preprocess/test_preprocess.py
#python3 "$SCRIPT_DIR"/internal/test_internal.py
#python3 "$SCRIPT_DIR"/external/test_external.py
#python3 "$SCRIPT_DIR"/macro/test_macro.py
python3 -m pytest "$SCRIPT_DIR"/cuda_test -v
python3 -m pytest "$SCRIPT_DIR"/pyccel -v
python3 -m pytest "$SCRIPT_DIR"/epyccel -v -m "not parallel"
mpirun -n 4 python3 -m pytest "$SCRIPT_DIR"/epyccel/test_epyccel_mpi_modules.py -v

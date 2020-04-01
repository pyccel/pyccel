#!/bin/bash

INSTALL_DIR=$(python -c "import pyccel; print(pyccel.__path__[0])")
SITE_DIR=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf "import coverage; coverage.process_startup()" > ${SITE_DIR}/pyccel_cov.pth
printf "[run]\nparallel = True\nsource = ${INSTALL_DIR}\n[report]\ninclude = ${INSTALL_DIR}/*\n[xml]\noutput = cobertura.xml" > .coveragerc
export COVERAGE_PROCESS_START=$(pwd)/.coveragerc

python -m pytest tests -m "not parallel" --ignore=tests/ast --ignore=tests/printers --ignore=tests/symbolic
mpirun -n 4 python -m pytest tests/epyccel/test_epyccel_mpi_modules.py -v -m parallel
#mpirun -n 4 python -m pytest tests/epyccel -v -m parallel

coverage combine
coverage xml
rm ${SITE_DIR}/pyccel_cov.pth

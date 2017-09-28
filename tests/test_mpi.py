# coding: utf-8

#$ header procedure mpi_init()     results(int)
#$ header procedure mpi_finalize() results(int)
mpi_init     = eval('mpi_init')
mpi_finalize = eval('mpi_finalize')

ierr = mpi_init()
ierr = mpi_finalize()

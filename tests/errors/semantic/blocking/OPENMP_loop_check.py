# Statement after OMP_For_Loop must be a for loop.
# pylint: disable=missing-function-docstring, missing-module-docstring


#$ omp parallel

#$ omp for
x = 50

#$ omp end parallel

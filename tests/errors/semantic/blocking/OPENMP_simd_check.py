# Statement after OMP_Simd_Construct must be a for loop.
# pylint: disable=missing-function-docstring, missing-module-docstring


#$ omp parallel

#$ omp simd
x = 50

#$ omp end parallel

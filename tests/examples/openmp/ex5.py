# coding: utf-8

# This example is the python implementation of collapse.2.f from OpenMP 4.5 examples

#$ omp parallel
#$ omp do private(j,k) collapse(2) lastprivate(jlast, klast)
for k in range(1, 3):
    for j in range(1, 4):
        jlast=j
        klast=k
#$ omp end do

#$ omp single
print(klast, jlast)
#$ omp end single
#$ omp end parallel

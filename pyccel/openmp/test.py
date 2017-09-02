
#$ omp parallel num_threads(3)
#$ omp parallel default(private)
#$ omp parallel private(x,y)
#$ omp parallel proc_bind(master)

#$ omp do private(x)
#$ omp do firstprivate(y)
#$ omp do lastprivate(z)
#$ omp do reduction(+: x,y)
#$ omp do collapse(4)
#$ omp do ordered
#$ omp do ordered(4)
#$ omp do schedule(dynamic)

#$ omp do linear(j:1)

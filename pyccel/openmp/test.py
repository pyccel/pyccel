
#@ omp parallel num_threads(3)
#@ omp parallel default(private)
#@ omp parallel private(x,y)
#@ omp parallel proc_bind(master)

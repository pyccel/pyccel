# coding: utf-8

#$ omp parallel private(t_id)
nb_taches = thread_number()
t_id = thread_id()
print(("> thread  id     = ", t_id))
#$ omp end parallel

print(("> threads number = ", nb_taches))

# Pyccel OpenMP usage

## Using the Runtime Library Routines

OpenMP Runtime Library Routines for Pyccel work by importing the OpenMP routine needed from the Pyccel stdlib:

```python
from pyccel.stdlib.internal.openmp import omp_set_num_threads
```
#### Example :
```python
from pyccel.decorators import types

@types(int)
def set_num_threads(n):
    from pyccel.stdlib.internal.openmp import omp_set_num_threads
    omp_set_num_threads(n)
```

## Directives Usage on Pyccel
### Parallel Constructs

#### Syntax :

```python
#$ omp parallel [clause[ [,] clause] ... ]
  structured-block
#$ omp end parallel
```
#### Example :

```python
from pyccel.stdlib.internal.openmp import omp_get_num_threads
#$ omp parallel
n = omp_get_num_threads()
#$ omp end parallel
```

### Loop Constructs

#### Syntax :

```python
#$ omp for [clause[ [,] clause] ... ]
  for-loops
```
#### Example :

```python
result = 0
#$ omp parallel private(i)
#$ omp for reduction (+:result)
for i in range(0, 1000):
  result += i
#$ omp end parallel
```

### Single Constructs

#### Syntax :

```python
#$ omp single [clause[ [,] clause] ... ]
  structured-block
#$ omp end single [end_clause[ [,] end_clause] ... ]
```
#### Example :

```python
result = 0
#$ omp parallel
#$ omp single
for i in range(0, 1000):
  result += i
#$ omp end single
#$ omp end parallel
```

### simd Constructs

#### Syntax :

```python
#$ omp simd [clause[ [,] clause] ... ]
  for-loops
```
#### Example :

```python
#$ omp simd
for i in range(0, 1000):
  result[i] = i
```

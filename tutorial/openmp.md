# Pyccel OpenMP usage

## Using the Runtime Library Routines

OpenMP Runtime Library Routines for Pyccel work by importing the OpenMP routine needed from the Pyccel stdlib:

```python
from pyccel.stdlib.internal.openmp import omp_set_num_threads
```

### Example

```python
from pyccel.decorators import types

@types(int)
def set_num_threads(n):
    from pyccel.stdlib.internal.openmp import omp_set_num_threads
    omp_set_num_threads(n)
```

## Directives Usage on Pyccel

### Parallel Construct

#### Syntax

```python
#$ omp parallel [clause[ [,] clause] ... ]
  structured-block
#$ omp end parallel
```

#### Example

```python
from pyccel.stdlib.internal.openmp import omp_get_num_threads
#$ omp parallel
n = omp_get_num_threads()
#$ omp end parallel
```

### Loop Construct

#### Syntax

```python
#$ omp for [clause[ [,] clause] ... ]
  for-loops
```

#### Example

```python
result = 0
#$ omp parallel private(i)
#$ omp for reduction (+:result)
for i in range(0, 1000):
  result += i
#$ omp end parallel
```

### Single Construct

#### Syntax

```python
#$ omp single [clause[ [,] clause] ... ]
  structured-block
#$ omp end single [end_clause[ [,] end_clause] ... ]
```

#### Example

```python
result = 0
#$ omp parallel
#$ omp single
for i in range(0, 1000):
  result += i
#$ omp end single
#$ omp end parallel
```

### Teams Construct

#### Syntax

```python
#$ omp teams [clause[ [,]clause] ... ]
  structured-block
#$ omp end teams
```

#### Example

```python
from pyccel.stdlib.internal.openmp import omp_get_team_num, omp_get_num_teams
result0 = 0
result1 = 0
nteams = 2
#$ omp teams num_teams(nteams)
tm_id = omp_get_team_num();
if omp_get_num_teams() == 2:
  if tm_id == 0:
    #$ omp parallel
    #$ omp for reduction (+:result0)
    for i in range(0, 1000):
       result0 += i
    #$ omp end parallel
  if tm_id == 1:
    #$ omp parallel
    #$ omp for reduction (+:result1)
    for i in range(0, 5000):
       result1 += i
    #$ omp end parallel
#$ omp end teams
result = result1 + result2
```

### Target Construct

#### Syntax

```python
#$  omp target [clause[ [,]clause] ... ]
  structured-block
#$ omp end target
```

#### Example

```python

#$ omp target
#$ omp parallel
#$ omp for private(i)
for i in range(0, 1000):
    result[i] = v1[i] * v2[i];
#$ omp end parallel
#$ omp end target
```

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
for i in range(0, 1337):
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
for i in range(0, 1337):
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
    for i in range(0, 1337):
       result0 += i
    #$ omp end parallel
  if tm_id == 1:
    #$ omp parallel
    #$ omp for reduction (+:result1)
    for i in range(0, 1337):
       result1 += i
    #$ omp end parallel
#$ omp end teams
result = result1 + result2
```

### Target Construct

#### Syntax

```python
#$ omp target [clause[ [,]clause] ... ]
  structured-block
#$ omp end target
```

#### Example

```python

#$ omp target
#$ omp parallel
#$ omp for private(i)
for i in range(0, 1337):
    result[i] = v1[i] * v2[i]
#$ omp end parallel
#$ omp end target
```

### Critical Construct

#### Syntax

```python
#$ omp critical [(name) [ [,] hint (hint-expression)]]
  structured-block
#$ omp end critical
```

#### Example

```python
result = 0
#$ omp parallel num_threads(4) shared(result)
for i in range(0, 1337):
  #$ omp critical
  result += i
  #$ omp end critical
#$ omp end parallel
```

### Barrier Construct

#### Syntax

```python
#$ omp barrier
```

#### Example

```python

#$ omp parallel
#$ omp for private(i)
for i in range(0, 1337):
  result[i] = v1[i] * v2[i]
#$ omp barrier
work(result)
#$ omp end parallel
```

### Atomic Construct

#### Syntax

```python
#$ omp atomic [clause[ [,]clause] ... ]
  structured-block
#$ omp end atomic
```

#### Example

```python

#$ omp parallel shared(result)
#$ omp for
for i in range(0, N):
  #$ omp atomic
  result = result + 1
  #$ omp end atomic
#$ omp end parallel
```

### Masked Construct

#### Syntax

```python
#$ omp masked [ filter(integer-expression) ]
  structured-block
#$ omp end masked
```

#### Example

```python
result = 0
#$ omp parallel shared(result)
#$ omp masked
result = result + 1
#$ omp end masked
#$ omp end parallel
```

### Task / Taskwait Construct

#### Syntax Task Construct

```python
#$ omp task [clause[ [,]clause] ... ]
  structured-block
#$ omp end task
```

#### Syntax Taskwait Construct

```python
#$ omp taskwait
```

#### Example

```python
@types('int')
def fib(n):
  if n < 2:
    return n
  #$ omp task shared(i) firstprivate(n)
  i = fib(n-1)
  #$ omp end task
  #$ omp task shared(j) firstprivate(n)
  j = fib(n-2)
  #$ omp end task
  #$ omp taskwait
  return i+j

#$ omp parallel
#$ omp omp single
result = fib(42)
#$ omp end single
#$ omp end parallel
```

### Taskyield Construct

#### Syntax

```python
#$ omp taskyield
```

#### Example

```python
#$ omp task
long_function()
#pragma omp taskyield
long_function_2()
#$ omp end task
```

### Flush Construct

#### Syntax

```python
#$ omp flush
```

#### Example

```python
from pyccel.stdlib.internal.openmp import omp_get_thread_num
flag = 0
#$ omp parallel num_threads(2)
if omp_get_thread_num() == 0:
  data = 1337
  #$ omp flush(flag, data)
  flag = 1
  #$ omp flush(flag)
elif omp_get_thread_num() == 1:
  #$ omp flush(flag, data)
  while flag < 1:
    #$ omp flush(flag, data)
  #$ flush(flag, data)
#$ omp end parallel
```

### Cancel Construct

#### Syntax

```python
#$ omp cancel construct-type-clause[ [ , ] if-clause]
```

#### Example

```python
result = 0
#$ omp parallel
#$ omp for private(i) reduction (+:result)
for i in range(len(v)):
  result = result + v[i]
  if result < 0:
    #$ omp cancel for
#$ omp end parallel
```

### SIMD Construct

#### Syntax

```python
#$ omp simd [clause[ [,]clause] ... ]
  loop-nest
```

#### Example

```python
#$ omp parallel
#$ omp simd private(i)
for i in range(N):
  result[i] = i
#$ omp end parallel
```

### Sections Worksharing Construct

#### Syntax

```python
#$ omp sections [clause[ [,]clause] ... ]
#$ omp section
  structured-block-sequence
#$ omp end section
#$ omp section
  structured-block-sequence
#$ omp end section
#$ omp end sections
```

#### Example

```python
section_count = 0
#$ omp parallel num_threads(2)
#$ omp omp sections firstprivate( section_count )

#$ omp section
section_count = section_count + 1
#$ omp end section

#$ omp section
section_count = section_count + 1
#$ omp end section
#$ omp omp end sections

#$ omp end parallel
```

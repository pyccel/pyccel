# Pyccel OpenMP usage

## Using the Runtime Library Routines

OpenMP Runtime Library Routines for Pyccel work by importing the OpenMP routine needed from the Pyccel stdlib:

Please note that files using the OpenMP Runtime library routines will only work when compiled with pyccel (i.e. they won't work in pure python mode).

```python
from pyccel.stdlib.internal.openmp import omp_set_num_threads
```

### Example

The following example shows how ``` omp_set_num_threads ``` is used to set the number of threads to ``` 4 threads ``` and how ``` omp_get_num_threads ``` is used to get the number of threads in the current team within a parallel region; ``` omp_get_num_threads ``` will return ``` 4 threads ```.

```python
from pyccel.decorators import types

@types('int')
def get_num_threads(n):
    from pyccel.stdlib.internal.openmp import omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
    omp_set_num_threads(n)
    #$ omp parallel
    print("hello from thread number:", omp_get_thread_num())
    result = omp_get_num_threads()
    #$ omp end parallel
    return result
x = get_num_threads(4)
print(x)
```
Please note that the variable ``` result ``` is a shared variable; Pyccel considers all variables as shared unless you specify them as private using the ``` private() ``` clause.

The output of this program is (you may get different result because of threads running at the same time):
```shell
❯ pyccel omp_test.py --openmp
❯ ./prog_omp_test
hello from thread number: 0
hello from thread number: 2
hello from thread number: 1
hello from thread number: 3
4
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

The following example shows how to use the ``` #$ omp parallel ``` pragma to create a team of 2 threads, each thread with its own private copy of the variables ``` n ```.

```python
from pyccel.stdlib.internal.openmp import omp_get_thread_num

#$ omp parallel private (n) num_threads(2)
n = omp_get_thread_num()
print("hello from thread:", n)
#$ omp end parallel
```

The output of this program is (you may get different result because of threads running at the same time):
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
hello from thread: 0
hello from thread: 1
```

### Loop Construct

#### Syntax

```python
#$ omp for [clause[ [,] clause] ... ]
for-loops
```

#### Example

This example shows how we can use the ``` #$ omp for ``` pragma to specify the loop that we want to be executed in parallel; each iteration of the loop is executed by one of the threads in the team.\
The ``` reduction ``` clause is used to deal with the data race, each thread has its own local copy of the reduction variable ``` result ```, when the threads join together, all the local copies of the reduction variable are combined to the global shared variable.

```python
result = 0
#$ omp parallel private(i) shared(result) num_threads(4)
#$ omp for reduction (+:result)
for i in range(0, 1337):
  result += i
#$ omp end parallel
print(result)
```

The output of this program is:
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
893116
```

### Single Construct

#### Syntax

```python
#$ omp single [clause[ [,] clause] ... ]
structured-block
#$ omp end single [end_clause[ [,] end_clause] ... ]
```

#### Example

This example shows how we can use the ``` #$ omp single ``` pragma to specify a section of code that must be run by a single available thread.

```python
from pyccel.stdlib.internal.openmp import omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
omp_set_num_threads(4)
#$ omp parallel
print("hello from thread number:", omp_get_thread_num())
#$ omp single
print("The best thread is number : ", omp_get_thread_num())
#$ omp end single
#$ omp end parallel
```

The output of this program is (you may get different result because of threads running at the same time):
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
hello from thread number:            1
The best thread is number :             1
hello from thread number:            2
hello from thread number:            3
hello from thread number:            0
```

### Critical Construct

#### Syntax

```python
#$ omp critical [(name) [ [,] hint (hint-expression)]]
structured-block
#$ omp end critical
```

#### Example

This example shows how ``` #$ omp critical ``` is used to specify the code which must be executed by one thread at a time.
```python
sum = 0
#$ omp parallel num_threads(4) private(i) shared(sum)
#$ omp for
for i in range(0, 1337):
  #$ omp critical
  sum += i
  #$ omp end critical
#$ omp end parallel
print(sum)
```

The output of this program is:
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
893116
```

### Barrier Construct

#### Syntax

```python
#$ omp barrier
```

#### Example

This example shows how ``` #$ omp barrier ``` is used to specify a point in the code where each thread must wait until all threads in the team arrive.
```python
from numpy import zeros

n = 1337
arr = zeros((n))
arr_2 = zeros((n))
#$ omp parallel num_threads(4) private(i, j) shared(arr)

#$ omp for
for i in range(0, n):
  arr[i] = i
#$ omp barrier
#$ omp for
for j in range(0, n):
  arr_2[j] = arr[j] * 2

#$ omp end parallel
print(sum(arr_2))
```

The output of this program is:
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
1786232
```

### Masked Construct

#### Syntax

```python
#$ omp masked [ filter(integer-expression) ]
structured-block
#$ omp end masked
```

#### Example

The ``` #$ omp masked ``` pragma is used here to specify a structured block that is executed by a subset of the threads of the current team.
```python
result = 0
#$ omp parallel num_threads(4)
#$ omp masked
result = result + 1
#$ omp end masked
#$ omp end parallel
print("result :", result)
```

The output of this program is:
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
result : 1
```

### Taskloop/Atomic Construct

#### Syntax Taskloop

```python
#$ omp taskloop [clause[ [,]clause] ... ]
for-loops
```

#### Syntax Atomic

```python
#$ omp atomic [clause[ [,]clause] ... ]
structured-block
#$ omp end atomic
```

#### Example
The ``` #$ omp taskloop ``` construct specifies that the iterations of one or more associated loops will be executed in parallel using explicit tasks.\
The ``` #$ omp atomic ``` is used to ensure that a specific storage location is accessed atomically.
```python
from pyccel.stdlib.internal.openmp import omp_get_thread_num

x1 = 0
x2 = 0
#$ omp parallel shared(x1,x2) num_threads(2)

#$ omp taskloop
for i in range(0, 100):
  #$ omp atomic
  x1 = x1 + 1 #Will be executed (100 x 2) times.
  #$ omp end atomic

#$ omp masked
#$ omp taskloop
for i in range(0, 100):
  #$ omp atomic
  x2 = x2 + 1 #Will be executed (100) times.
  #$ omp end atomic
#$ omp end masked

#$ omp end parallel
print("x1 : ", x1);
print("x2 : ", x2);
```

The output of this program is (you may get a different output, but the sum must be the same for each thread):
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
x1 : 200
x2 : 100
```

### SIMD Construct

#### Syntax

```python
#$ omp simd [clause[ [,]clause] ... ]
loop-nest
```

#### Example

The ``` #$ omp simd ``` pragma is used to transform the loop into a loop that will be executed concurrently using Single Instruction Multiple Data (SIMD) instructions.
```python
from numpy import zeros
result = 0
n = 1337
arr = zeros(n, dtype=int)
#$ omp parallel num_threads(4)
#$ omp simd
for i in range(0, n):
  arr[i] = i
#$ omp end parallel
for i in range(0, n):
  result = result + arr[i]
print("Result:", result)
```

The output of this program is:
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
Result: 893116
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

The ``` #$ omp task ``` pragma is used here to define an explicit task.\
The ``` #$ omp taskwait ``` pragma is used here to specify a wait on the completion of child tasks of the current task.
```python
@types('int', results='int')
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
print(fib(10))
#$ omp end single
#$ omp end parallel

print("result :", result)
```

The output of this program is:
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
55
```

### Taskyield Construct

#### Syntax

```python
#$ omp taskyield
```

#### Example

The ``` #$ omp taskyield ``` pragma specifies that the current task can be suspended at this point, in favor of execution of a different task.

```python
#$ omp task
long_function()
#$ omp taskyield
long_function_2()
#$ omp end task
```

### Flush Construct

#### Syntax

```python
#$ omp flush
```

#### Example

The ``` #$ omp flush ``` pragma is used to ensure that all threads in a team have a consistent view of certain objects in memory.
```python
from pyccel.stdlib.internal.openmp import omp_get_thread_num
flag = 0
#$ omp parallel num_threads(2)
if omp_get_thread_num() == 0:
  #$ omp atomic update
  flag = flag + 1
elif omp_get_thread_num() == 1:
  #$ omp flush(flag)
  while flag < 1:
    #$ omp flush(flag)
  print("Thread 1 released")
  #$ omp atomic update
  flag = flag + 1
#$ omp end parallel
print("flag:", flag)
```

The output of this program is:
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
Thread 1 released
flag: 2
```

### Cancel Construct

#### Syntax

```python
#$ omp cancel construct-type-clause[ [ , ] if-clause]
```

#### Example

The ``` #$ omp cancel ``` is used to request cancellation of the innermost enclosing region of the type specified.
```python
result = 0
#$ omp parallel
#$ omp for private(i) reduction (+:result)
for i in range(len(v)):
  result = result + v[i]
  if result < 0:
    break
    #$ omp cancel for
#$ omp end parallel
```

### Teams/Target/distribute Constructs

#### Syntax Teams Constructs

```python
#$ omp teams [clause[ [,]clause] ... ]
structured-block
#$ omp end teams
```

#### Syntax Target Constructs

```python
#$ omp target [clause[ [,]clause] ... ]
structured-block
#$ omp end target
```

#### Syntax Distribute Constructs

```python
#$ omp distribute [clause[ [,]clause] ... ]
for-loops
```

#### Example

In this example we show how we can use the ``` #$ omp target ``` pragma to define a target region, which is a computational block that operates within a distinct data environment and is intended to be offloaded onto a parallel computation device during execution.\
The ``` #$ omp teams ``` directive creates a collection of thread teams. The master thread of each team executes the teams region.\
The ``` #$ omp distribute ``` directive specifies that the iterations of one or more loops will be executed by the thread teams in the context of their implicit tasks.
```python
from numpy import zeros
from pyccel.stdlib.internal.openmp import omp_get_team_num
n = 8
a = zeros(n, dtype=int)
#$ omp target map(to: n) map(tofrom: a)
#$ omp teams num_teams(2) thread_limit(n/2)
#$ omp distribute
for i in range(0, n):
  a[i] = omp_get_team_num()
#$ omp end teams
#$ omp end target
for i in range(0, n):
  print("Team num :", a[i])
```

The output of this program is:
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
Team num : 0
Team num : 0
Team num : 0
Team num : 0
Team num : 1
Team num : 1
Team num : 1
Team num : 1
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

The ``` #$ omp sections ``` directive is used to distribute work among threads (2 threads).

```python
from pyccel.stdlib.internal.openmp import omp_get_thread_num

n = 8
sum1 = 0
sum2 = 0
sum3 = 0
#$ omp parallel num_threads(2)
#$ omp omp sections

#$ omp section
for i in range(0, int(n/3)):
  sum1 = sum1 + i
print("sum1 :", sum1, ", thread :", omp_get_thread_num())
#$ omp end section

#$ omp section
for i in range(0, int(n/2)):
  sum2 = sum2 + i
print("sum2 :", sum2, ", thread :", omp_get_thread_num())
#$ omp end section

#$ omp section
for i in range(0, n):
  sum3 = sum3 + i
print("sum3 :", sum3, ", thread :", omp_get_thread_num())
#$ omp end section
#$ omp omp end sections

#$ omp end parallel
```

The output of this program is :
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
sum1 : 1, thread : 0
sum2 : 6, thread : 0
sum3 : 28, thread : 1
```

# Pyccel OpenMP usage

## Using the Runtime Library Routines

OpenMP Runtime Library Routines for Pyccel work by importing the OpenMP routine needed from the `pyccel.stdlib`:

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

### Supported Routines

From the many routines defined in the [OpenMP 5.1 Standard](https://www.openmp.org/spec-html/5.1/openmp.html), Pyccel currently supports:

-   All thread team routines except ``` omp\_get\_supported\_active\_levels ```
-   All thread affinity routines except ``` omp\_set\_affinity\_format ```, ``` omp\_get\_affinity\_format ```, ``` omp\_display\_affinity ```, ``` omp\_capture\_affinity ```
-   All tasking routines
-   All device information routines except ``` omp\_get\_device\_num ```
-   `omp\_get\_num\_teams`
-   `omp\_get\_team\_num`

## Directives Usage on Pyccel

Pyccel uses the same clauses as OpenMP, you can refer to the references below for more information on how to use them:

[*OpenMP 5.1 API Specification (pdf)*](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-1.pdf)\
[*OpenMP 5.1 API Specification (html)*](https://www.openmp.org/spec-html/5.1/openmp.html)
[*OpenMP 5.1 Syntax Reference Guide*](https://www.openmp.org/wp-content/uploads/OpenMPRefCard-5.1-web.pdf)

Other references:

[*OpenMP Clauses*](https://docs.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-clauses)

### parallel Construct

#### Syntax of *parallel*

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

### loop Construct

#### Syntax of *loop*

```python
#$ omp for [nowait] [clause[ [,] clause] ... ]
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

### single Construct

#### Syntax of *single*

```python
#$ omp single [nowait] [clause[ [,] clause] ... ]
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

### critical Construct

#### Syntax of *critical*

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

### barrier Construct

#### Syntax of *barrier*

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

### masked Construct

#### Syntax of *masked*

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

### `taskloop`/`atomic` Construct

#### Syntax of *`taskloop`*

```python
#$ omp taskloop [clause[ [,]clause] ... ]
for-loops
```

#### Syntax of *atomic*

```python
#$ omp atomic [clause[ [,]clause] ... ]
structured-block
#$ omp end atomic
```

#### Example
The ``` #$ omp taskloop ``` construct specifies that the iterations of one or more associated loops will be executed in parallel using explicit tasks.\
The ``` #$ omp atomic ``` is used to ensure that a specific storage location is accessed atomically; which prevent the possibility of multiple, simultaneous reading and writing of threads.
```python
from pyccel.stdlib.internal.openmp import omp_get_thread_num

x1 = 0
x2 = 0
#$ omp parallel shared(x1,x2) num_threads(2)

#$ omp taskloop
for i in range(0, 100):
  #$ omp atomic
  x1 = x1 + 1 #Will be executed (100 x 2) times.

#$ omp single
#$ omp taskloop
for i in range(0, 100):
  #$ omp atomic
  x2 = x2 + 1 #Will be executed (100) times.
#$ omp end single

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

### simd Construct

#### Syntax of *simd*

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

### task / taskwait Construct

#### Syntax of *task*

```python
#$ omp task [clause[ [,]clause] ... ]
structured-block
#$ omp end task
```

#### Syntax *taskwait*

```python
#$ omp taskwait
```

#### Example

The ``` #$ omp task ``` pragma is used here to define an explicit task.\
The ``` #$ omp taskwait ``` pragma is used here to specify that the current task region remains suspended until all child tasks that it generated before the taskwait construct complete execution.
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
#$ omp single
print(fib(10))
#$ omp end single
#$ omp end parallel
```

The output of this program is:
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
55
```

### taskyield Construct

#### Syntax of *taskyield*

```python
#$ omp taskyield
```

#### Example

The ``` #$ omp taskyield ``` pragma specifies that the current task can be suspended at this point, in favour of execution of a different task.

```python
#$ omp task
long_function()
#$ omp taskyield
long_function_2()
#$ omp end task
```

### flush Construct

#### Syntax of *flush*

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

### cancel Construct

#### Syntax of *cancel*

```python
#$ omp cancel construct-type-clause[ [ , ] if-clause]
```

#### Example

The ``` #$ omp cancel ``` is used to request cancellation of the innermost enclosing region of the type specified.
```python
import numpy as np
v = np.array([1, -5, 3, 4, 5])
result = 0
#$ omp parallel
#$ omp for private(i) reduction (+:result)
for i in range(len(v)):
  result = result + v[i]
  if result < 0:
    #$ omp cancel for
    pass
#$ omp end parallel
```

### teams/target/distribute Constructs

#### Syntax *target*

```python
#$ omp target [clause[ [,]clause] ... ]
structured-block
#$ omp end target
```

#### Syntax of *teams*

```python
#$ omp teams [clause[ [,]clause] ... ]
structured-block
#$ omp end teams
```

#### Syntax *distribute*

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
threadlimit = 4
a = zeros(n, dtype=int)
#$ omp target
#$ omp teams num_teams(2) thread_limit(threadlimit)
#$ omp distribute
for i in range(0, n):
  a[i]    = omp_get_team_num()
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

### sections Construct

#### Syntax of *sections*

```python
#$ omp sections [nowait] [clause[ [,]clause] ... ]

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
#$ omp sections

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

#$ omp end sections
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

## Combined Constructs Usage on Pyccel

### parallel for

#### Syntax of *parallel for*

```python
#$ omp parallel for [clause[ [,]clause] ... ]
loop-nest
```

#### Example

The ```#$ omp parallel for``` construct specifies a parallel construct containing a worksharingloop construct with a canonical loop nest.

```python
import numpy as np
x = np.array([2,5,4,3,2,5,7])
result = 0
#$ omp parallel for reduction (+:result)
for i in range(0, len(x)):
    result += x[i]
print("result:", result)
```

The output of this program is :
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
result: 28
```

### parallel for simd

#### Syntax of *parallel for simd*

```python
#$ omp parallel for simd [clause[ [,]clause] ... ]
loop-nest
```

#### Example

The ```#$ omp parallel for simd``` construct specifies a parallel construct containing only one worksharing-loop SIMD construct.

```python
import numpy as np
x = np.array([1,2,1,2,1,2,1,2])
y = np.array([2,1,2,1,2,1,2,1])
z = np.zeros(8, dtype = int)
result = 0
#$ omp parallel for simd
for i in range(0, 8):
    z[i] = x[i] + y[i]

for i in range(0, 8):
    print("z[",i,"] :", z[i])
```

The output of this program is :
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
z[ 0 ] : 3
z[ 1 ] : 3
z[ 2 ] : 3
z[ 3 ] : 3
z[ 4 ] : 3
z[ 5 ] : 3
z[ 6 ] : 3
z[ 7 ] : 3
```
### for simd

#### Syntax of *for simd*

```python

#$ omp for simd [clause[ [,]clause] ... ]
for-loops
```

### teams distribute

#### Syntax of *teams distribute*

```python
#$ omp teams distribute [clause[ [,]clause] ... ]
loop-nest
```

#### Example

```python
import numpy as np
x = np.array([1,2,1,2,1,2,1,2])
y = np.array([2,1,2,1,2,1,2,1])
z = np.zeros(8, dtype = int)
result = 0
#$ omp parallel
#$ omp for simd
for i in range(0, 8):
    z[i] = x[i] + y[i]

#$ omp end parallel
for i in range(0, 8):
    print("z[",i,"] :", z[i])
```

The output of this program is :
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
z[ 0 ] : 3
z[ 1 ] : 3
z[ 2 ] : 3
z[ 3 ] : 3
z[ 4 ] : 3
z[ 5 ] : 3
z[ 6 ] : 3
z[ 7 ] : 3
```


### teams distribute simd

#### Syntax of *teams distribute simd*

```python
#$ omp teams distribute simd [clause[ [,]clause] ... ]
loop-nest
```

### teams distribute parallel for

#### Syntax of *teams distribute parallel for*

```python
#$ omp teams distribute parallel for [clause[ [,]clause] ... ]
loop-nest
```

### target parallel

#### Syntax of *target parallel*

```python
#$ omp target parallel [clause[ [,]clause] ... ]
structured-block
#$ omp end target parallel
```

### target parallel for

#### Syntax of *target parallel for*

```python
#$ omp target parallel for [clause[ [,]clause] ... ]
loop-nest
```

### target parallel for simd

#### Syntax of *target parallel for simd*

```python
#$ omp target parallel for simd [clause[ [,]clause] ... ]
loop-nest
```

### target teams

#### Syntax of *target teams*

```python
#$ omp target teams [clause[ [,]clause] ... ]
structured-block
#$ omp end target teams
```

### target teams distribute

#### Syntax of *target teams distribute*

```python
#$ omp target teams distribute [clause[ [,]clause] ... ]
loop-nest
```

### target teams distribute simd

#### Syntax of *target teams distribute simd*

```python
#$ omp target teams distribute simd [clause[ [,]clause] ... ]
loop-nest
```

### target teams distribute parallel for

#### Syntax of *target teams distribute parallel for*

```python
#$ omp target teams distribute parallel for [clause[ [,]clause] ... ]
loop-nest
```

### target teams distribute parallel for simd

#### Syntax of *target teams distribute parallel for simd*

```python
#$ omp target teams distribute parallel for simd [clause[ [,]clause] ... ]
loop-nest
```

#### Example

The ```#$ omp parallel for simd``` construct specifies a parallel construct containing only one worksharing-loop SIMD construct.

```python
r = 0
#$ omp target teams distribute parallel for reduction(+:r)
for i in range(0, 10000):
    r = r + i

print("result:",r)
```

The output of this program is :
```shell
❯ pyccel omp_test.py --openmp
❯ ./omp_test
result: 49995000
```

## Supported Constructs

All constructs in the OpenMP 5.1 standard are supported except:
-   `scope`
-   `workshare`
-   `scan`
-   `interop`

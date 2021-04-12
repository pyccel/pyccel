# task-parallelism with Pyccel

## A task

Parallel program is viewed as a collection of tasks that communicate by sending
messages to each other through dependencies.
A task consists of an executable unit or a unit of computation (think of it as a program),
together with its local memory and a collection of I/O ports. that can/should execute in
parallel with other tasks.

## Context

A problem can be borken into multiple parts each part can be represented as a task, each
task is a separate unit of work that can be independant or may take some dependencies from other
tasks. task with dependencies may only start when all antecedents have completed. 
So in general the dependencies between tasks can be represented as a directed acyclic graph
where tasks form the vertices edge are the dependencies between task.

## Usage:
Using the decorator task allows pyccel to recognize that the current function will
be used in task parallelism and automatically calculate its dependencies.

```Python
@task
def A():
    # do something
```

A task function can only run in parallel if the parent function is a task.
```Python
@task('master')
def main():
    a = A()
```

```Python
@task('master')
def recursive() -> int:
    a = recursive()
    # do something
```


A block of tasks that need to run in parallel is by the first occurrence of a task 
function in (FunctionCall or Assign) until the first non task block of the end of the program.
So avoid writing a codeblock that depend on a task result between tasks calls.

```Python
@task('master')
def main():
    x, y = # do something
    
    # this will run in parallel
    # ------
    a = A(x) # first task A
    b = B(y) # second task B
    # ------

    c = a + b # first non task block

```
Explaining dependencies :

```Python
@task
def A(x : int):
    return x

@task
def B(x : int):
    return x

@task
def C(x : int, y : int):
    return x + y


@task('master')
def main():
    a = A(1)
    b = B(1)
    c = C(a + b)
```
dependencies graph :
A----
     |---- C
B----

In this scenario C cannot run until A and B are completed, and A, B will run in parallel.

```Python
@task
def main():
    a = A(1)
    b = B(1)
    c = C(a + b)
    d = D(a)
```

dependencies graph:
A---- ------D
     |------C
B----

in this scenario A, B will run in parallel, C will wait for then to complets, and D will only wait for A to complete.
So D can run in parallel with C

```Python
@task
def main():
    a = A(1)
    d = D(a)
    e = E(1)
```

dependencies graph:
A -----------> D
E


in this scenario D will wait for A to complet, but E is independant so it can run in parallel with A or D (depend on available threads)


Currently complex manipulations of task are not completly supported (in automatic generation) yet like :
- task as if condition                       ```if(x > A(2))```
- task as a for loop condition               ```for i in range(A(2))```
- task as function or another task argument  ```A(B(1))```
- task as index                              ```arr[A(1)]```
- task in operations                         ```a = A(1) + 2```

so the best practice will be to avoid any undefined behavior by storing the task result in a variable and us it freely.

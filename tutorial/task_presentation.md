# task-parallelism with Pyccel

## A task

Parallel program is viewed as a collection of tasks that communicate by sending
messages to each other through dependencies.
A task consists of an executable unit or a unit of computation (think of it as a program),
together with its local memory and a collection of I/O ports. that can/should execute in
parallel with other tasks.

## Context

A problem can be broken into multiple parts each part can be represented as a task, each
task is a separate unit of work that can be independant or may take some dependencies from other
tasks. task with dependencies may only start when all antecedents have completed. 
So in general the dependencies between tasks can be represented as a directed acyclic graph
where tasks form the vertices edge are the dependencies between task.
Usage:
using the decorator task allow pyccel to recognize that the current function will be used
in task paralism and automaticly calcule its dependencies.
A task function can only be parallized if the parrent function is a task, this will allow
pyccel to know when function need to be parallized.

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
@task 
def D(x : int):
    return x
@task 
def E(x : int):
    print(x)
@task
def main():
    a = A(1)
    b = B(1)
    c = C(a + b)
```

- dependencies graph :

`
A----

        |---- C
B----
`

In this scenario C cannot run until A and B are complets, and A, B will run in parallel.

```Python
@task
def main():
    a = A(1)
    b = B(1)
    c = C(a + b)
    d = D(a)
```

- dependencies graph:
`
A----
      ------D

     |------C
B----
`

in this scenario A, B will run in parallel, C will wait for them to complet, and D will only wait for A to complete.
So D can run in parallel with C

```Python
@task
def main():
    a = A(1)
    d = D(a)
    e = E(1)
```

- dependencies graph:
`
A -----------> D

E
`

in this scenario D will wait for A to complet, but E is independant so it can't run in parallel with A or D (depend on available threads)
Currently complex manipulations of task are not completly supported (in automatic generation) yet like :

- task as if condition
```if(x > A(2))```

- task as a for loop condition
```for i in range(A(2))```

- task as function or another task argument
```A(B(1))```

- task as index
```arr[A(1)]```

- task in operations
```a = A(1) + 2```

so the best practice will be to avoid any undefined behavior by storing the task result in a variable and us it freely.

avoid writing a codeblock that depend on a task result between tasks calls

```Python
@task
def main():
    a = A(1)
    if a > 0:
        # do something
    b = B(1)
```

in this case parallel is not effective because the if condition should wait for A to finish
so A and B can't run in parallel.

OpenMP
******

We follow `OpenMP 4.5`_ specifications.

.. _OpenMP 4.5: http://www.openmp.org/wp-content/uploads/openmp-4.5.pdf

Grammar
^^^^^^^

In this section, we give the **BNF** used grammar for parsing *openmp*.

.. literalinclude:: ../../pyccel/parser/grammar/openmp.tx 

See :download:`script <../../pyccel/parser/grammar/openmp.tx>`.


Directives
^^^^^^^^^^

OpenMP directives for Fortran are specified as follows::

  sentinel directive-name [clause[ [,] clause]...]

The following sentinels are recognized in fixed form source files::

  !$omp | c$omp | *$omp

Constructs
^^^^^^^^^^

parallel
________

The syntax of the parallel construct is as follows::

  !$omp parallel [clause[ [,] clause] ... ]
    structured-block
  !$omp end parallel

where *clause* is one of the following::

  if([parallel :] scalar-logical-expression)
  num_threads(scalar-integer-expression)
  default(private | firstprivate | shared | none)
  private(list)
  firstprivate(list)
  shared(list)
  copyin(list)
  reduction(reduction-identifier : list)
  proc_bind(master | close | spread)

The **end parallel** directive denotes the end of the **parallel** construct.

.. todo:: add restrictions (page 49)

Loop
____

The syntax of the loop construct is as follows::

  !$omp do [clause[ [,] clause] ... ]
    do-loops
  [!$omp end do [nowait]]

where *clause* is one of the following::

  private(list)
  firstprivate(list)
  lastprivate(list)
  linear(list[ : linear-step])
  reduction(reduction-identifier : list)
  schedule([modifier [, modifier]:]kind[, chunk_size])
  collapse(n)
  ordered[(n)]

If an **end do** directive is not specified, an **end do** directive is assumed at the end of the do-loops.

sections
________

The syntax of the sections construct is as follows::

  !$omp sections [clause[ [,] clause] ... ]
    [!$omp section]
      structured-block
    [!$omp section
      structured-block]
    ...
  !$omp end sections [nowait]

where *clause* is one of the following::

  private(list)
  firstprivate(list)
  lastprivate(list)
  reduction(reduction-identifier : list)

single
______

The syntax of the single construct is as follows::

  !$omp single [clause[ [,] clause] ... ]
    structured-block
  !$omp end single [end_clause[ [,] end_clause] ... ]

where *clause* is one of the following::

  private(list)
  firstprivate(list)

and *end_clause* is one of the following::

  copyprivate(list)
  nowait

workshare
_________

The syntax of the workshare construct is as follows::

  !$omp workshare
    structured-block
  !$omp end workshare [nowait]

The enclosed structured block must consist of only the following::

  array assignments
  scalar assignments
  FORALL statements
  FORALL constructs
  WHERE statements
  WHERE constructs
  atomic constructs
  critical constructs
  parallel constructs

simd
____

The syntax of the simd construct is as follows::

  !$omp simd [clause[ [,] clause ... ]
    do-loops
  [!$omp end simd]

where *clause* is one of the following::

  safelen(length)
  simdlen(length)
  linear(list[ : linear-step])
  aligned(list[ : alignment])
  private(list)
  lastprivate(list)
  reduction(reduction-identifier : list)
  collapse(n)

If an **end simd** directive is not specified, an **end simd** directive is assumed at the end of the *do-loops*.

declare simd
____________

The syntax of the declare simd construct is as follows::

  !$omp declare simd [(proc-name)] [clause[ [,] clause] ... ]

where *clause* is one of the following::

  simdlen(length)
  linear(linear-list[ : linear-step])
  aligned(argument-list[ : alignment])
  uniform(argument-list)
  inbranch
  notinbranch

Loop simd
_________

The syntax of the Loop simd construct is as follows::

  !$omp do simd [clause[ [,] clause] ... ]
    do-loops
  [!$omp end do simd [nowait] ]

where *clause* can be any of the clauses accepted by the **simd** or **do** directives, with identical meanings and restrictions.

If an **end do simd** directive is not specified, an **end do simd** directive is assumed at the end of the do-loops.

.. todo:: finish the specs and add more details.

Data-Sharing Attribute Clauses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

default
_______

The syntax of the **default** clause is as follows::

  default(private | firstprivate | shared | none)

shared
_______

The syntax of the **shared** clause is as follows::

  shared(list)

private
_______

The syntax of the **private** clause is as follows::

  private(list)

firstprivate
____________

The syntax of the **firstprivate** clause is as follows::

  firstprivate(list)

lastprivate
___________

The syntax of the **lastprivate** clause is as follows::

  lastprivate(list)

reduction
_________

The syntax of the **reduction** clause is as follows::

  reduction(reduction-identifier : list)

linear
______

The syntax of the **linear** clause is as follows::

  linear(linear-list[ : linear-step])

where *linear-list* is one of the following::

  list
  modifier(list)

where *modifier* is one of the following::

  val

Data Copying Clauses
^^^^^^^^^^^^^^^^^^^^

copyin
______

The syntax of the **copyin** clause is as follows::

  copyin(list)

copyprivate
___________

The syntax of the **copyprivate** clause is as follows::

  copyprivate(list)

Data-mapping Attribute Rules and Clauses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

map
___

The syntax of the *map* clause is as follows::

  map([ [map-type-modifier[,]] map-type : ] list)

where *map-type* is one of the following::

  to
  from
  tofrom
  alloc
  release
  delete

and *map-type-modifier* is **always**.

defaultmap
__________

The syntax of the *defaultmap* clause is as follows::

  defaultmap(tofrom:scalar)

declare reduction Directive
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The syntax of the *declare reduction* directive is as follows::

  !$omp declare reduction(reduction-identifier : type-list : combiner)
  [initializer-clause]

where:

1. *reduction-identifier* is either a base language identifier, or a user-defined operator, or one of the following operators: **+**, **-**, ***** , **.and.**, **.or.**, **.eqv.**, **.neqv.**, or one of the following intrinsic procedure names: **max**, **min**, **iand**, **ior**, **ieor**.
2. *type-list* is a list of type specifiers
3. *combiner* is either an assignment statement or a subroutine name followed by an argument list
4. *initializer-clause* is **initializer** (*initializer-expr*), where *initializer-expr* is **omp_priv** = *expression* or *subroutine-name* (*argument-list*)

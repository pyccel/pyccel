Validation
**********

OK tests
^^^^^^^^

ex2, ex3, ex5,   

KO tests
^^^^^^^^

* ex1::

    ('> uncovered statement of type : ', <class 'pyccel.ast.core.ThreadsNumber'>)
    ('> uncovered statement of type : ', <class 'pyccel.ast.core.ThreadID'>)


* ex4::

    File "/home/macahr/projects/pyccel/pyccel/parser/syntax/core.py", line 554, in get_arguments_zeros
    raise TypeError('Unexpected type')
  
* ex6::

    textx.exceptions.TextXSyntaxError: Expected '=' or '=' or '=' or ',' or '=' or '=' or '(' or '[' or '.' or '=' or '(' or '[' or '.' or '+=' or '*=' or '-=' or '/=' or ',' or '=' at position (21, 9) => 't         *#$ omp end'.

* ex7::

    File "/home/macahr/projects/pyccel/pyccel/parser/syntax/core.py", line 986, in expr_with_trailer
    raise NotImplementedError('Only FunctionDef is treated')
    NotImplementedError: Only FunctionDef is treated

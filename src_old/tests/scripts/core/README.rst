KO
**

* the following instruction is not working (ex6.py)::
    
    f = e[0,2]
    print(f)

  **f** is then declared as a *pointer*::

  .. code-block:: fortran

    real(kind=8), pointer :: f

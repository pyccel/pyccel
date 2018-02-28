Rules
*****

In this section, we provide a set of rules that were used to ensure the **one-to-one** correspondance between *Python* and *Fortran*. These rules are applied while annotating the *AST*.

Warnings
^^^^^^^^

- **WF001**: all returned arguments must be atoms, no expression is allowed

- **WF002**:   

Errors
^^^^^^

- **EF001**: a function with with at least one argument or returned value, must have an associated header 


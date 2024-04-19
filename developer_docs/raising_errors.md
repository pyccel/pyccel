# Raising Errors

In Pyccel we strive to raise readable errors wherever possible. The errors must provide the user with enough information to find and correct the issue in their code. Additionally they should also be able to provide developers with enough information to find and debug related errors. Finally we also want users to be able to catch errors emitted by `epyccel` or `lambdify` to decide how to handle these errors.

In order to handle these different interests Pyccel has several classes related to errors. These can be found in the folder `pyccel/errors/`.

## Types of errors

Pyccel generates 3 different kinds of errors:
-   `PyccelSyntaxError`
-   `PyccelSemanticError`
-   `PyccelCodegenError`


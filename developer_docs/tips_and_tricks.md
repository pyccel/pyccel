# Tips and Tricks

-   While _pytest_ gives lots of information about the source of bugs, it can be a bit overwhelming. The `pyccel` command line tool gives you the most readable set of information about the source of errors and makes it easy to check that the generated code looks as expected
-   When using the `pyccel` command line tool, use the flag `--developer-mode` to see tracebacks for the user-readable error messages
-   When using `pytest` the flag `--developer-mode` can also be used to see useful tracebacks for non-fatal error messages.
-   The environment variable `PYCCEL_ERROR_MODE` can be used to set the default error mode for Pyccel (`export PYCCEL_ERROR_MODE=developer`).
-   When trying to find a bug where the generated code gives different results to the original Python code, a good first step is usually to generate the corresponding code and try to find where the implementation differs
-   When there is a problem in the compile stage the first step must **always** be to run `pyccel` with the `--verbose` flag and test the generated compile commands manually. It is impossible to fix the compile problem without knowing what the correct command should be
-   When you have broken previously working code, comparing the difference in the generated code before and after your changes should help you target which of your changes caused the problems
-   Sometimes the traceback provided by Pyccel in developer-mode is too short to find the problem. If this is the case, the length of the traceback can be changed temporarily in `pyccel.errors.errors.Errors.report` (the default traceback length is 5).

Please feel free to add any tips or tricks you find to this non-exhaustive list

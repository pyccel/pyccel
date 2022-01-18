# Tips and Tricks

-   While _pytest_ gives lots of information about the source of bugs, it can be a bit overwhelming. The `pyccel` command line tool gives you the most readable set of information about the source of errors and makes it easy to check that the generated code looks as expected
-   When using the `pyccel` command line tool, use the flag `--developer-mode` to see tracebacks for the (fatal) user-readable error messages
-   When trying to find a bug where the generated code gives different results to the original Python code, a good first step is usually to generate the corresponding code and try to find where the implementation differs
-   When there is a problem in the compile stage the first step must **always** be to run `pyccel` with the `--verbose` flag and test the generated compile commands manually. It is impossible to fix the compile problem without knowing what the correct command should be

_Please feel free to add any tips or tricks you find to this non-exhaustive list_

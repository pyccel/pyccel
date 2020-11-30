# Header files

## Using header files

A header file in Pyccel is a file with a name ending with `.pyh`, which contains function/variable declarations, macro definitions, templates and metavariable declarations.\
Header files serve two purposes:
-   Link external libraries in the targeted languages by providing their function declarations;
-   Accelerate the parsing process of an imported Python module by parsing only its header file (automatically generated) instead of the full module.

### Example
We create the file `header.pyh` that contains an openmp function definition:

```python
#$ header metavar module_name = 'omp_lib'
#$ header metavar import_all  = True

#$ header function omp_get_num_threads() results(int)
```
We then create `openmp.py` file:

```python
from header import omp_get_num_threads
print('number of threads is :', omp_get_num_threads())
```
Pyccel can compile the Python file with the following command: `pyccel openmp.py --language fortran --flags="-fopenmp"`
, It will then create the executable file `openmp`

## Pickling header files
Parsing a large Pyccel header file with hundreds of function declarations may require a significant amount of time, therefore it is important that this process is only done once when pyccelizing multiple Python source files in a large project.

To this end, Pyccel uses the [pickle](https://docs.python.org/3/library/pickle.html) Python module to store the result of the parser to a `.pyccel` binary file, which is created in the same directory as the header file.
Afterwards Pyccel will load the precompiled parser from the `.pyccel` file, instead of parsing the header file again.
This results in a performance gain.

Pyccel will generate a new `.pyccel` binary if the corresponding header file was modified, or if the installed version of Pyccel does not match the one used to parse the header.

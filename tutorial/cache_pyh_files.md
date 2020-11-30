# Picklize header files

## header files

A header file in pyccel is a file containing function or variable declarations, macro definitions , templates and metavariable declarations.
the convention is to give header files names that end with `.pyh` .
Header files serve two purposes:
- link externel libraries in the targeted languages by providing their function definitions.
- accelerate the parssing process by parsing the header file instead of the original file in the case of pyccelizing multiple files.

### Example
We create the file `header.pyh` that contains an openmp function definiton:

```python
#$ header metavar module_name = 'omp_lib'
#$ header metavar import_all  = True

#$ header function omp_get_num_threads() results(int)
```
We then create `openmp.py` file that contains:

```python
from header import omp_get_num_threads
print(omp_get_num_threads())
```
Pyccel can compile the python file with the following command: `pyccel openmp.py --language fortran --flags="-fopenmp"`
it will then create the executable file `openmp`

## picklizing the header file
When compiling a header file pyccel will generate in the same directory `.pyccel` file that contains the cached result of the parser, 
This will accelerate the compiling process of big header files, by compiling them only once and storing the results for futur compilation, Pyccel will generate a new
`.pyccel` when changing the `.pyh` or downloading a new version of Pyccel


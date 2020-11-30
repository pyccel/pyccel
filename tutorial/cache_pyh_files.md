# Picklize Header files

## header files

A header file in pyccel is a file containing function/variable declarations, macro definitions , templates and metavariable declarations.\
the rule is to give header files names that end with `.pyh` .\
Header files serve two purposes:
-Link external libraries in the targeted languages by providing their function definitions.
-Accelerate the parsing process by parsing the header file instead of the original file in the case of pyccelizing multiple files.

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
print('number of threads is :', omp_get_num_threads())
```
Pyccel can compile the python file with the following command: `pyccel openmp.py --language fortran --flags="-fopenmp"`
, It will then create the executable file `openmp`

## picklizing the header file
Pyccel uses the Python Module [pickle](https://docs.python.org/3/library/pickle.html) to cache the header files.\
When compiling a header file pyccel will generate in the same directory a `.pyccel` file that contains the cached result of the parser,\
This will accelerate the compiling process of big header files, by compiling them only once and storing the results for future compilation, Pyccel will generate a new\
`.pyccel` when changing the `.pyh` or downloading a new version of Pyccel.

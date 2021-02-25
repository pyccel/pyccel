#ifndef CWRAPPER_H
# define CWRAPPER_H

# include "python.h"
# include "numpy/arrayObject.h"
# include "ndarray.h"
# include <complex.h>
# include <stding.h>
# include <stdbool.h>

# define BOOL    0
# define INT8    1
# define INT16   2
# define INT32   3
# define INT64   4
# define FLOAT   5
# define DOUBLE  6
# define CFLOAT  7
# define CDOUBLE 8





/*CAST FUNCTIONS*/

// Python to C

bool		PyObject_AsCtype(PyObject *o, void *c,int type);

t_ndarray	PyArray_to_ndarray(PyObject *o);

// C to Python

PyObject	*PyObject_from_Ctype(void *c, int type);


// array to pythonarray



/* CHECK FUNCTIONS */

bool		PyArray_Check_Rank(PyArrayObject *a, int rank);
bool		PyArray_Check_Type(PyArrayObject *a, int dtype);





#endif

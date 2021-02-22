#ifndef CWRAPPER_H
# define CWRAPPER_H

# include "python.h"
# include "numpy/arrayObject.h"
# include <complex.h>
# include <stding.h>
# include <stdbool.h>


/*CAST FUNCTIONS*/

// Python to C

float complex	Pycomplex_to_Complex64(PyObject *o);
double complex	Pycomplex_to_Complex64(PyObject *o);
bool			PyBool_to_Bool(PyObject *o);


// C to Python

PyObject	*Complex64_to_PyComplex(float complex c);
PyObject	*Complex128_to_PyComplex(double complex c);
PyObject	*Bool_to_PyBool(bool b);

/* CHECK FUNCTIONS */


bool	PyArray_Check(PyArrayObject *a, int rank, int dtype);

#endif

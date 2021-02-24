#ifndef CWRAPPER_H
# define CWRAPPER_H

# include "python.h"
# include "numpy/arrayObject.h"
# include "ndarray.h"
# include <complex.h>
# include <stding.h>
# include <stdbool.h>


/*CAST FUNCTIONS*/

// Python to C

int			Pycomplex_to_Complex64(PyObject *o, float complex *c);
int			Pycomplex_to_Complex128(PyObject *o, double complex *c);

int			PyInt64_to_Int64(PyObject *o, int64_t *i);
int			PyInt32_to_Int32(PyObject *o, int32_t *i);
int			PyInt16_to_Int16(PyObject *o, int16_t *i);
int			PyInt8_to_Int8(PyObject *o, int8_t *i);

int			Pyfloat_to_Float(PyObject *o, float *f);
int			Pydouble_to_Double(PyObject *o, double *f);

int			PyBool_to_Bool(PyObject *o, bool *b);

t_ndarray	PyArray_to_ndarray(PyObject *o);

// C to Python

PyObject	*Complex64_to_PyComplex(float complex c);
PyObject	*Complex128_to_PyComplex(double complex *c);

PyObject	*Int64_to_PyInt64(int64_t *i);
PyObject	*Int32_to_PyInt32(int32_t *i);
PyObject	*Int16_to_PyInt16(int16_t *i);
PyObject	*Int8_to_PyInt8(int8_t *i);

PyObject	*Float_to_PyFloat(int8_t *i);
PyObject	*Double_to_PyDouble(int8_t *i);

PyObject	*Bool_to_PyBool(int8_t *i);

// array to pythonarray



/* CHECK FUNCTIONS */

bool		PyArray_Check_Rank(PyArrayObject *a, int rank);
bool		PyArray_Check_Type(PyArrayObject *a, int dtype);





#endif

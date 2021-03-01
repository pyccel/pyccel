#ifndef CWRAPPER_H
# define CWRAPPER_H



# include "Python.h"
# include "numpy/arrayobject.h"
# include <complex.h>
# include <stdint.h>
# include <stdbool.h>

/* functions prototypes */

/* casting python object to c type */
float complex	Pycomplex_to_Complex64(PyObject *o, float complex *c);
double complex	Pycomplex_to_Complex128(PyObject *o, double complex *c);

int64_t			PyInt64_to_Int64(PyObject *o);
int32_t			PyInt32_to_Int32(PyObject *o);
int16_t			PyInt16_to_Int16(PyObject *o);
int8_t			PyInt8_to_Int8(PyObject *o);

float			Pyfloat_to_Float(PyObject *o);
double			Pydouble_to_Double(PyObject *o);

bool			PyBool_to_Bool(PyObject *o);

/* numpy array to ndarray */
t_ndarray	PyArray_to_ndarray(PyObject *o);


/* casting c type to python object */
PyObject	*Complex_to_PyComplex(double complex c);

PyObject	*Bool_to_PyBool(bool b);

PyObject	*Int_to_PyLong(int64_t i);

PyObject	*Double_to_PyDouble(double d);

/* array check */

bool			PyArray_CheckType(PyArrayObject *a, int dtype);
PyArrayObject	*Check_Array(PyObject *a, int rank, int flags);



#endif

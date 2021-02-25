#ifndef CWRAPPER_H
# define CWRAPPER_H

# include "python.h"
# include "numpy/arrayObject.h"
# include "ndarray.h"
# include <complex.h>
# include <stding.h>
# include <stdbool.h>



/* functions prototypes */

/* casting python object to c type */
bool		Pycomplex_to_Complex64(PyObject *o, float complex *c);
bool		Pycomplex_to_Complex128(PyObject *o, double complex *c);

bool		PyInt64_to_Int64(PyObject *o, int64_t *i);
bool		PyInt32_to_Int32(PyObject *o, int32_t *i);
bool		PyInt16_to_Int16(PyObject *o, int16_t *i);
bool		PyInt8_to_Int8(PyObject *o, int8_t *i);

bool		Pyfloat_to_Float(PyObject *o, float *f);
bool		Pydouble_to_Double(PyObject *o, double *f);

bool		PyBool_to_Bool(PyObject *o, bool *b);

/* numpy array to ndarray */
t_ndarray	PyArray_to_ndarray(PyObject *o);


/* casting c type to python object */
PyObject	*Complex64_to_PyComplex(float complex *c;
PyObject	*Complex128_to_PyComplex(double complex *c);

PyObject	*Bool_to_PyBool(bool *b);

PyObject	*Int64_to_PyInt64(int64_t *i);

PyObject	*Double_to_PyDouble(double *d);



/* array check */

bool		PyArray_Check_Rank(PyArrayObject *a, int rank);
bool		PyArray_Check_Type(PyArrayObject *a, int dtype);


#endif

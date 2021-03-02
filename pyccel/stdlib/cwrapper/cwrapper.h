#ifndef CWRAPPER_H
# define CWRAPPER_H
# define PY_SSIZE_T_CLEAN

# include "Python.h"
# include "numpy/arrayobject.h"
# include <complex.h>
# include <stdint.h>
# include <stdbool.h>
# include "ndarrays.h"
# define NO_TYPE_CHECK -1
# define NO_ORDER_CHECK -1


// strings order needs to be the same as its equivalent numpy macro
// https://numpy.org/doc/stable/reference/c-api/dtype.html
const char* dataTypes[17] = {"Bool", "Int8", "UInt8", "Int16", "UIn16", "Int32", "UInt32",
                             "Int64", "UInt64", "Int128", "UInt128", "Float32", "Float64",
                             "Float128", "Complex64", "Complex128", "Complex256"};


/* array converter */

bool	pyarray_to_ndarray(PyObject *o, t_ndarray *array, int dtype, int rank, int flag)

/* functions prototypes */

/* casting python object to c type */
float complex	Pycomplex_to_Complex64(PyObject *o, float complex *c);
double complex	Pycomplex_to_Complex128(PyObject *o, double complex *c);

int64_t			PyInt64_to_Int64(PyObject *o);
int32_t			PyInt32_to_Int32(PyObject *o);
int16_t			PyInt16_to_Int16(PyObject *o);
int8_t			PyInt8_to_Int8(PyObject *o);

float			PyFloat_to_Float(PyObject *o);
double			PyDouble_to_Double(PyObject *o);

bool			PyBool_to_Bool(PyObject *o);

/* numpy array to ndarray */
//t_ndarray	PyArray_to_ndarray(PyArrayObject *o);


/* casting c type to python object */
PyObject	*Complex_to_PyComplex(double complex c);

PyObject	*Bool_to_PyBool(bool b);

PyObject	*Int_to_PyLong(int64_t i);

PyObject	*Double_to_PyDouble(double d);


#endif

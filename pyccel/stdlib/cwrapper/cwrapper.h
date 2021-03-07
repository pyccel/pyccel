#ifndef CWRAPPER_H
# define CWRAPPER_H
# define PY_SSIZE_T_CLEAN

# include "Python.h"
# include <complex.h>
# include <stdint.h>
# include <stdbool.h>

# define NO_IMPORT_ARRAY
# define PY_ARRAY_UNIQUE_SYMBOL CWRAPPER_ARRAY_API
# include "numpy/arrayobject.h"

# define NO_TYPE_CHECK -1
# define NO_ORDER_CHECK -1




#ifdef NDARRAYS_H

# include "ndarrays.h"
t_ndarray	pyarray_to_ndarray(PyArrayObject *o);

#endif


/* array converter */
bool        pyarray_check(PyArrayObject *o, int dtype, int rank, int flag);
int32_t     array_ndim(PyArrayObject *o, int index);

/* functions prototypes */

/* casting python object to c type */
float complex	PyComplex_to_Complex64(PyObject *o) ;
double complex	PyComplex_to_Complex128(PyObject *o);

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
PyObject	*Complex128_to_PyComplex(double complex *c);
PyObject	*Complex64_to_PyComplex(float complex *c);

PyObject	*Bool_to_PyBool(bool *b);

PyObject	*Int64_to_PyLong(int64_t *i);
PyObject	*Int32_to_PyLong(int32_t *i);
PyObject	*Int16_to_PyLong(int16_t *i);
PyObject	*Int8_to_PyLong(int8_t *i);

PyObject	*Double_to_PyDouble(double *d);
PyObject	*Float_to_PyDouble(float *d);


#endif

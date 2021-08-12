/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

/*
 * File containing functions useful for the cwrapper.
 * There are 3 types of functions:
 * - Functions converting PythonObjects to standard C types
 * - Functions converting standard C types to PythonObjects
 * - Functions which test the type of PythonObjects
 */

#ifndef CWRAPPER_H
# define CWRAPPER_H
# define PY_SSIZE_T_CLEAN

# include "Python.h"
# include <complex.h>
# include <stdint.h>
# include <stdbool.h>
# include "numpy_version.h"

# define NO_IMPORT_ARRAY
# define PY_ARRAY_UNIQUE_SYMBOL CWRAPPER_ARRAY_API
# include "numpy/arrayobject.h"

# define NO_TYPE_CHECK -1
# define NO_ORDER_CHECK -1


/*
 * Function: pyarray_checker
 * --------------------
 * Check Python Object (DataType, Rank, Order):
 *
 * Parameters :
 *     a 	 : python array object
 *     dtype : desired data type enum
 *     rank  : desired rank
 *     flag  : desired order flag
 *
 * Returns	  :
 *     return true if no error occurred otherwise it will return false
 */
bool	pyarray_checker(PyArrayObject *o, int dtype, int rank, int flag);

/*
 * Functions : Cast functions
 * --------------------------
 * All functions listed down are based on C/python api
 * with more tolerance to different precision
 * Convert python type object to the desired C type
 * Parameters :
 *     object : the python object
 * Returns    :
 *     The desired C type, an error may be raised by c/python converter
 *     so one should call PyErr_Occurred() to check for errors after the
 *	   calling a cast function
 */
float complex	PyComplex_to_Complex64(PyObject *o) ;
double complex	PyComplex_to_Complex128(PyObject *o);

int64_t			PyInt64_to_Int64(PyObject *o);
int32_t			PyInt32_to_Int32(PyObject *o);
int16_t			PyInt16_to_Int16(PyObject *o);
int8_t			PyInt8_to_Int8(PyObject *o);

float			PyFloat_to_Float(PyObject *o);
double			PyDouble_to_Double(PyObject *o);

bool			PyBool_to_Bool(PyObject *o);


/*
 * Functions : Cast functions
 * ---------------------------
 * Some of the function used below are based on C/python api
 * with more tolerance to different precisions and complex type.
 * Collect the python object from the C object
 * Parameters :
 *     object : the C object
 *
 * Returns    :
 *     boolean : python object
 */
PyObject	*Complex128_to_PyComplex(double complex *c);
PyObject	*Complex64_to_PyComplex(float complex *c);

PyObject	*Bool_to_PyBool(bool *b);

PyObject	*Int64_to_PyLong(int64_t *i);
PyObject	*Int32_to_PyLong(int32_t *i);
PyObject	*Int16_to_PyLong(int16_t *i);
PyObject	*Int8_to_PyLong(int8_t *i);

PyObject	*Double_to_PyDouble(double *d);
PyObject	*Float_to_PyDouble(float *d);

/*
 * Functions : Type check functions
 * ---------------------------
 * Some of the function used below are based on C/python api and numpy/c api with
 * more tolerance to different precisions, different system architectures and complex type.
 * Check the C data type ob a python object
 * Parameters :
 *     object     : the python object
 *     hard_check : boolean true if intensive precision check is needed
 *
 * Returns    :
 *     boolean : logic statement responsible for checking python data type
 */
bool    PyIs_Int8(PyObject *o, bool hard_check);
bool    PyIs_Int16(PyObject *o, bool hard_check);
bool    PyIs_Int32(PyObject *o, bool hard_check);
bool    PyIs_Int64(PyObject *o, bool hard_check);
bool    PyIs_Float(PyObject *o, bool hard_check);
bool    PyIs_Double(PyObject *o, bool hard_check);
bool    PyIs_Bool(PyObject *o, bool hard_check);
bool    PyIs_Complex128(PyObject *o, bool hard_check);
bool    PyIs_Complex64(PyObject *o, bool hard_check);


#endif

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

extern const char* dataTypes[17];

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
 *
 * Reference of the used c python api function
 * --------------------------------------------
 * https://docs.python.org/3/c-api/float.html#c.PyFloat_AsDouble
 * https://docs.python.org/3/c-api/long.html#c.PyLong_AsLong
 * https://docs.python.org/3/c-api/long.html#c.PyLong_AsLongLong
 */
float complex	PyComplex_to_Complex64(PyObject *o) ;
double complex	PyComplex_to_Complex128(PyObject *o);

//-----------------------------------------------------//
static inline int64_t	PyInt64_to_Int64(PyObject *object)
{
	return (int64_t)PyLong_AsLongLong(object);
}
//-----------------------------------------------------//
static inline int32_t	PyInt32_to_Int32(PyObject *object)
{
	return (int32_t)PyLong_AsLong(object);
}
//-----------------------------------------------------//
static inline int16_t	PyInt16_to_Int16(PyObject *object)
{
	return (int16_t)PyLong_AsLong(object);
}
//-----------------------------------------------------//
static inline int8_t	PyInt8_to_Int8(PyObject *object)
{
	return (int8_t)PyLong_AsLong(object);
}
//-----------------------------------------------------//
static inline bool	PyBool_to_Bool(PyObject *object)
{
	return object == Py_True;
}
//-----------------------------------------------------//
static inline float	PyFloat_to_Float(PyObject *object)
{
	return (float)PyFloat_AsDouble(object);
}
//-----------------------------------------------------//
static inline double	PyDouble_to_Double(PyObject *object)
{
	return PyFloat_AsDouble(object);
}


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
PyObject	*Complex128_to_NumpyComplex(double complex *c);
PyObject	*Complex64_to_NumpyComplex(float complex *c);

PyObject	*Bool_to_PyBool(bool *b);

PyObject	*Int64_to_PyLong(int64_t *i);
PyObject	*Int32_to_PyLong(int32_t *i);
PyObject	*Int64_to_NumpyLong(int64_t *i);
PyObject	*Int32_to_NumpyLong(int32_t *i);
PyObject	*Int16_to_NumpyLong(int16_t *i);
PyObject	*Int8_to_NumpyLong(int8_t *i);

PyObject	*Double_to_PyDouble(double *d);
PyObject	*Double_to_NumpyDouble(double *d);
PyObject	*Float_to_NumpyDouble(float *d);

/*
 * Functions : Type check functions
 * ---------------------------
 * Some of the function used below are based on C/python api and numpy/c api with
 * more tolerance to different precisions, different system architectures and complex type.
 * Check the C data type ob a python object
 * Parameters :
 *     object     : the python object
 *
 * Returns    :
 *     boolean : logic statement responsible for checking python data type
 *
 * Reference of the used c/python api function
 * ---------------------------------------------------
 * https://docs.python.org/3/c-api/long.html#c.PyLong_Check
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_Check
 * https://docs.python.org/3/c-api/float.html#c.PyFloat_Check
 * https://docs.python.org/3/c-api/bool.html#c.PyBool_Check
 * https://numpy.org/doc/1.17/reference/c-api.array.html#c.PyArray_IsScalar
 */
//--------------------------------------------------------//
static inline bool    PyIs_NativeInt(PyObject *o)
{
    return PyLong_CheckExact(o);
}
//--------------------------------------------------------//
static inline bool    PyIs_Int8(PyObject *o)
{
    return PyArray_IsScalar(o, Int8);
}
//--------------------------------------------------------//
static inline bool    PyIs_Int16(PyObject *o)
{
    return PyArray_IsScalar(o, Int16);
}
//--------------------------------------------------------//
static inline bool    PyIs_Int32(PyObject *o)
{
    return PyArray_IsScalar(o, Int32);
}
//--------------------------------------------------------//
static inline bool    PyIs_Int64(PyObject *o)
{
    return PyArray_IsScalar(o, Int64);
}
//--------------------------------------------------------//
static inline bool    PyIs_NativeFloat(PyObject *o)
{
    return PyFloat_Check(o);
}
//--------------------------------------------------------//
static inline bool    PyIs_Float(PyObject *o)
{
    return PyArray_IsScalar(o, Float32);
}
//--------------------------------------------------------//
static inline bool    PyIs_Double(PyObject *o)
{
	return PyArray_IsScalar(o, Float64);
}
//--------------------------------------------------------//
static inline bool    PyIs_Bool(PyObject *o)
{
	return PyBool_Check(o) || PyArray_IsScalar(o, Bool);
}
//--------------------------------------------------------//
static inline bool    PyIs_NativeComplex(PyObject *o)
{
	return PyComplex_Check(o);
}
//--------------------------------------------------------//
static inline bool    PyIs_Complex128(PyObject *o)
{
	return PyArray_IsScalar(o, Complex128);
}
//--------------------------------------------------------//
static inline bool    PyIs_Complex64(PyObject *o)
{
    return PyArray_IsScalar(o, Complex64);
}


#endif

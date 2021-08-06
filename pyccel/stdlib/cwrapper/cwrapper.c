#include "cwrapper.h"



// strings order needs to be the same as its equivalent numpy macro
// https://numpy.org/doc/stable/reference/c-api/dtype.html
const char* dataTypes[17] = {"Bool", "Int8", "UInt8", "Int16", "UIn16", "Int32", "UInt32",
                             "Int64", "UInt64", "Int128", "UInt128", "Float32", "Float64",
                             "Float128", "Complex64", "Complex128", "Complex256"};



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
 * Reference of the used c python api function
 * --------------------------------------------
 * https://docs.python.org/3/c-api/float.html#c.PyFloat_AsDouble
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_RealAsDouble
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_ImagAsDouble
 * https://docs.python.org/3/c-api/long.html#c.PyLong_AsLong
 * https://docs.python.org/3/c-api/long.html#c.PyLong_AsLongLong
 */

float complex PyComplex_to_Complex64(PyObject *object)
{
	float complex	c;

	// https://numpy.org/doc/1.17/reference/c-api.array.html#c.PyArray_IsScalar
	// https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_ScalarAsCtype
	if (PyArray_IsScalar(object, Complex64))
		PyArray_ScalarAsCtype(object, &c);

	else
	{
		float real_part = (float)PyComplex_RealAsDouble(object);
		float imag_part = (float)PyComplex_ImagAsDouble(object);

		c = real_part + imag_part * _Complex_I;
	}
	return	c;
}
//-----------------------------------------------------//
double complex	PyComplex_to_Complex128(PyObject *object)
{
	double complex	c;
	double	real_part;
	double	imag_part;

	real_part = PyComplex_RealAsDouble(object);
	imag_part = PyComplex_ImagAsDouble(object);

	c = real_part + imag_part * _Complex_I;

	return	c;
}
//-----------------------------------------------------//
int64_t	PyInt64_to_Int64(PyObject *object)
{
	int64_t		i;
	long long	out;

	out = PyLong_AsLongLong(object);

	i = (int64_t)out;

	return	i;
}
//-----------------------------------------------------//
int32_t	PyInt32_to_Int32(PyObject *object)
{
	int32_t	i;
	long	out;

	out = PyLong_AsLong(object);

	i = (int32_t)out;

	return	i;
}
//-----------------------------------------------------//
int16_t	PyInt16_to_Int16(PyObject *object)
{
	int16_t	i;
	long	out;

	out = PyLong_AsLong(object);

	i = (int16_t)out;

	return	i;
}
//-----------------------------------------------------//
int8_t	PyInt8_to_Int8(PyObject *object)
{
	int8_t	i;
	long	out;

	out = PyLong_AsLong(object);

	i = (int8_t)out;

	return	i;
}
//-----------------------------------------------------//
bool	PyBool_to_Bool(PyObject *object)
{
	bool	b;

	b = object == Py_True;

	return	b;
}
//-----------------------------------------------------//
float	PyFloat_to_Float(PyObject *object)
{
	float	f;
	double	out;

	out = PyFloat_AsDouble(object);

	f = (float)out;

	return	f;
}
//-----------------------------------------------------//
double	PyDouble_to_Double(PyObject *object)
{
	double	d;

	d = PyFloat_AsDouble(object);

	return	d;
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
 * reference of the used c/python api function
 * ---------------------------------------------------
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_FromDoubles
 * https://docs.python.org/3/c-api/float.html#c.PyFloat_FromDouble
 * https://docs.python.org/3/c-api/long.html#c.PyLong_FromLongLong
 */

PyObject	*Complex128_to_PyComplex(double complex *c)
{
	double		real_part;
	double		imag_part;
	PyObject	*object;

	real_part = creal(*c);
	imag_part = cimag(*c);
	object = PyComplex_FromDoubles(real_part, imag_part);

	return object;
}
//-----------------------------------------------------//
PyObject	*Complex64_to_PyComplex(float complex *c)
{
	float		real_part;
	float		imag_part;
	PyObject	*object;

	real_part = crealf(*c);
	imag_part = cimagf(*c);
	object = PyComplex_FromDoubles((double) real_part, (double) imag_part);

	return object;
}
//-----------------------------------------------------//
PyObject	*Bool_to_PyBool(bool *b)
{
	return *b == true ? Py_True : Py_False;
}
//-----------------------------------------------------//
PyObject	*Int64_to_PyLong(int64_t *i)
{
	PyObject	*object;

	object = PyLong_FromLongLong((long long) *i);

	return object;
}
//-----------------------------------------------------//
PyObject	*Int32_to_PyLong(int32_t *i)
{
	PyObject	*object;

	object = PyLong_FromLongLong((long long) *i);

	return object;
}
//-----------------------------------------------------//
PyObject	*Int16_to_PyLong(int16_t *i)
{
	PyObject	*object;

	object = PyLong_FromLongLong((long long) *i);

	return object;
}
//--------------------------------------------------------//
PyObject	*Int8_to_PyLong(int8_t *i)
{
	PyObject	*object;

	object = PyLong_FromLongLong((long long) *i);

	return object;
}
//--------------------------------------------------------//
PyObject	*Double_to_PyDouble(double *d)
{
	PyObject	*object;

	object = PyFloat_FromDouble(*d);

	return object;
}
//--------------------------------------------------------//
PyObject	*Float_to_PyDouble(float *d)
{
	PyObject	*object;

	object = PyFloat_FromDouble((double)*d);

	return object;
}

/*
 * Functions : Cast functions
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
 * Reference of the used c/python api function
 * ---------------------------------------------------
 * https://docs.python.org/3/c-api/long.html#c.PyLong_Check
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_Check
 * https://docs.python.org/3/c-api/float.html#c.PyFloat_Check
 * https://docs.python.org/3/c-api/bool.html#c.PyBool_Check
 * https://numpy.org/doc/1.17/reference/c-api.array.html#c.PyArray_IsScalar
 */

bool    PyIs_Int8(PyObject *o, bool hard_check)
{
	if (hard_check == true)
		return PyArray_IsScalar(o, Int8);

	return PyLong_Check(o) || PyArray_IsScalar(o, Int8);
}
//--------------------------------------------------------//
bool    PyIs_Int16(PyObject *o, bool hard_check)
{	
	if (hard_check == true)
		return PyArray_IsScalar(o, Int16);

	return PyLong_Check(o) || PyArray_IsScalar(o, Int16);
}
//--------------------------------------------------------//
bool    PyIs_Int32(PyObject *o, bool hard_check)
{
	#ifdef _WIN32
		return PyLong_Check(o) || PyArray_IsScalar(o, Int32);
    #else
        if (hard_check == true)
            return PyArray_IsScalar(o, Int32);

        return PyLong_Check(o) || PyArray_IsScalar(o, Int32);
	#endif
}
//--------------------------------------------------------//
bool    PyIs_Int64(PyObject *o, bool hard_check)
{
	#ifndef _WIN32
		return PyLong_Check(o) || PyArray_IsScalar(o, Int64);
    #else
        if (hard_check == true)
            return PyArray_IsScalar(o, Int64);

        return PyLong_Check(o) || PyArray_IsScalar(o, Int64);
	#endif
}
//--------------------------------------------------------//
bool    PyIs_Float(PyObject *o, bool hard_check)
{
	if (hard_check == true)
		return PyArray_IsScalar(o, Float32);

	return PyFloat_Check(o) || PyArray_IsScalar(o, Float32);
}
//--------------------------------------------------------//
bool    PyIs_Double(PyObject *o, bool hard_check)
{
	(void)hard_check;

	return PyFloat_Check(o) || PyArray_IsScalar(o, Float64);
}
//--------------------------------------------------------//
bool    PyIs_Bool(PyObject *o, bool hard_check)
{
	(void)hard_check;

	return PyBool_Check(o) || PyArray_IsScalar(o, Bool);
}
//--------------------------------------------------------//
bool    PyIs_Complex128(PyObject *o, bool hard_check)
{
	(void)hard_check;

	return PyComplex_Check(o) || PyArray_IsScalar(o, Complex64);
}
//--------------------------------------------------------//
bool    PyIs_Complex64(PyObject *o, bool hard_check)
{
	if (hard_check == true)
		return PyArray_IsScalar(o, Complex64);

	return PyComplex_Check(o) || PyArray_IsScalar(o, Complex64);
}

/*
 * Function: _check_pyarray_dtype
 * --------------------
 * Check Python Object DataType:
 *
 * Parameters :
 *     a 	 : python array object
 *     dtype : desired data type enum
 *
 * Returns	  :
 *		return true if no error occurred otherwise it will return false
 *      and raise TypeError exception
 *
 * Reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_TYPE
 * https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Format
 */
bool	check_pyarray_dtype(PyArrayObject *a, int dtype)
{
	int current_dtype;

	if (dtype == NO_TYPE_CHECK)
		return true;

	current_dtype = PyArray_TYPE(a);
	if (current_dtype != dtype)
	{
		PyErr_Format(PyExc_TypeError,
			"argument dtype must be %s, not %s",
			dataTypes[dtype],
			dataTypes[current_dtype]);
		return false;
	}

	return true;
}

/*
 * Function: _check_pyarray_rank
 * --------------------
 * Check Python Object Rank:
 *
 * Parameters :
 *     a 	  : python array object
 *     rank  : desired rank
 * Returns    :
 *     return true if no error occurred otherwise it will return false
 *     and raise TypeError exception
 *
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_NDIM
 * https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Format
 */
static bool _check_pyarray_rank(PyArrayObject *a, int rank)
{
	int	current_rank;

	current_rank = PyArray_NDIM(a);
	if (current_rank != rank)
	{
		PyErr_Format(PyExc_TypeError, "argument rank must be %d, not %d",
			rank,
			current_rank);
		return false;
	}

	return true;
}

/*
 * Function: _check_pyarray_order
 * --------------------
 * Check Python Object Order:
 *
 * Parameters	:
 *     a 	  : python array object
 *     flag  : desired order
 * Returns		:
 *     return true if no error occurred otherwise it will return false
 *     and raise NotImplementedError exception
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_CHKFLAGS
 * https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Format

 */
static bool _check_pyarray_order(PyArrayObject *a, int flag)
{

	if (flag == NO_ORDER_CHECK)
		return true;

	if (!PyArray_CHKFLAGS(a, flag))
	{
		char order = flag == NPY_ARRAY_C_CONTIGUOUS ? 'C' : 'F';
		PyErr_Format(PyExc_NotImplementedError,
			"argument does not have the expected ordering (%c)", order);
		return false;
	}

	return true;
}


/*
 * Function: _check_pyarray_type
 * --------------------
 * Check if Python Object is ArrayType:
 *
 * Parameters :
 *     a : python array object
 *
 * Returns   :
 *     return true if no error occurred otherwise it will return false
 *     and raise TypeError exception
 * Reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Check
 * https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Format
 */
static bool _check_pyarray_type(PyObject *a)
{
	if (!PyArray_Check(a))
	{
		PyErr_Format(PyExc_TypeError,
			"argument must be numpy.ndarray, not %s",
			 a == Py_None ? "None" : Py_TYPE(a)->tp_name);
		return false;
	}

	return true;
}

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
bool	pyarray_checker(PyArrayObject *o, int dtype, int rank, int flag)
{
	if (!_check_pyarray_type((PyObject *)o)) return false;

	// check array element type / rank / order
	if(!check_pyarray_dtype(o, dtype)) return false;

	if(!_check_pyarray_rank(o, rank)) return false;

	if(rank > 1 && !_check_pyarray_order(o, flag)) return false;

	return true;
}

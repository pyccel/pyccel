/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#include "cwrapper.h"



// strings order needs to be the same as its equivalent numpy macro
// https://numpy.org/doc/stable/reference/c-api/dtype.html
const char* dataTypes[17] = {"Bool", "Int8", "UInt8", "Int16", "UIn16", "Int32", "UInt32",
                             "Int64", "UInt64", "Int128", "UInt128", "Float32", "Float64",
                             "Float128", "Complex64", "Complex128", "Complex256"};




/* Casting python object to c type
 *
 * Reference of the used c python api function
 * --------------------------------------------
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_RealAsDouble
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_ImagAsDouble
 */
float complex PyComplex_to_Complex64(PyObject *object)
{
	float complex	c;

	// https://numpy.org/doc/1.17/reference/c-api.array.html#c.PyArray_IsScalar
	// https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_ScalarAsCtype
	if (PyArray_IsScalar(object, Complex64))
    {
		PyArray_ScalarAsCtype(object, &c);
    }
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
	double	real_part;
	double	imag_part;

	real_part = PyComplex_RealAsDouble(object);
	imag_part = PyComplex_ImagAsDouble(object);

	return real_part + imag_part * _Complex_I;
}


/* casting c type to python object
 *
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

	real_part = creal(*c);
	imag_part = cimag(*c);
	return PyComplex_FromDoubles(real_part, imag_part);
}
//-----------------------------------------------------//
PyObject	*Complex64_to_PyComplex(float complex *c)
{
	float		real_part;
	float		imag_part;

	real_part = crealf(*c);
	imag_part = cimagf(*c);
	return PyComplex_FromDoubles((double) real_part, (double) imag_part);
}
//-----------------------------------------------------//
PyObject	*Bool_to_PyBool(bool *b)
{
	return (*b) ? Py_True : Py_False;
}
//-----------------------------------------------------//
PyObject	*Int64_to_PyLong(int64_t *i)
{
	return PyLong_FromLongLong((long long) *i);
}
//-----------------------------------------------------//
PyObject	*Int32_to_PyLong(int32_t *i)
{
	return PyLong_FromLongLong((long long) *i);
}
//-----------------------------------------------------//
PyObject	*Int16_to_PyLong(int16_t *i)
{
	return PyLong_FromLongLong((long long) *i);
}
//--------------------------------------------------------//
PyObject	*Int8_to_PyLong(int8_t *i)
{
	return PyLong_FromLongLong((long long) *i);
}
//--------------------------------------------------------//
PyObject	*Double_to_PyDouble(double *d)
{
	return PyFloat_FromDouble(*d);
}
//--------------------------------------------------------//
PyObject	*Float_to_PyDouble(float *d)
{
	return PyFloat_FromDouble((double)*d);
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

/* arrays check*/
bool	pyarray_checker(PyArrayObject *o, int dtype, int rank, int flag)
{
	if (!_check_pyarray_type((PyObject *)o)) return false;

	// check array element type / rank / order
	if(!check_pyarray_dtype(o, dtype)) return false;

	if(!_check_pyarray_rank(o, rank)) return false;

	if(rank > 1 && !_check_pyarray_order(o, flag)) return false;

	return true;
}

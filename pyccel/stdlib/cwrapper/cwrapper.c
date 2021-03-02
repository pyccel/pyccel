#include "cwrapper.h"

/*
 * Functions : Cast functions
 * --------------------------
 * All functions listed down are based on C/python api
 * with more tolerance to different precision
 * Convert python type object to the desired C type
 * 	Parameters :
 *		o	: the python object
 *	Returns    :
 * 		The desired C type, an error may be raised by c/python converter
 *      so one should call PyErr_Occurred() to check for errors after the
 *		calling a cast function
 * reference of the used c python api function
 * --------------------------------------------
 * https://docs.python.org/3/c-api/float.html#c.PyFloat_AsDouble
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_RealAsDouble
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_ImagAsDouble
 * https://docs.python.org/3/c-api/long.html#c.PyLong_AsLong
 * https://docs.python.org/3/c-api/long.html#c.PyLong_AsLongLong
 */

float complex PyComplex_to_Complex64(PyObject *o)
{
	float complex	c;
	float			real_part;
	float			imag_part;


	real_part = (float)PyComplex_RealAsDouble(o);
	imag_part = (float)PyComplex_ImagAsDouble(o);

	c = CMPLXF(real_part, imag_part);

	return	c;
}

double complex	PyComplex_to_Complex128(PyObject *o)
{
	double complex	c;
	double			real_part;
	double			imag_part;

	real_part = PyComplex_RealAsDouble(o);
	imag_part = PyComplex_ImagAsDouble(o);

	c = CMPLX(real_part, imag_part);

	return	c;
}

int64_t	PyInt64_to_Int64(PyObject *o)
{
	int64_t		i;
	long long	out;

	out = PyLong_AsLongLong(o);

	i = (int64_t)out;

	return	i;
}

int32_t	PyInt32_to_Int32(PyObject *o)
{
	int32_t	i;
	long	out;

	out = PyLong_AsLong(o);

	i = (int32_t)out;

	return	i;
}

int16_t	PyInt16_to_Int16(PyObject *o)
{
	int16_t	i;
	long	out;

	out = PyLong_AsLong(o);

	i = (int16_t)out;

	return	i;
}

int8_t	PyInt8_to_Int8(PyObject *o)
{
	int8_t	i;
	long	out;

	out = PyLong_AsLong(o);

	i = (int8_t)out;

	return	i;
}

bool	PyBool_to_Bool(PyObject *o)
{
	bool	b;

	b = o == Py_True;

	return	b;
}

float	PyFloat_to_Float(PyObject *o)
{
	float	f;
	double	out;

	out = PyFloat_AsDouble(o);

	f = (float)out;

	return	f;
}

double	PyDouble_to_Double(PyObject *o)
{
	double	d;

	d = PyFloat_AsDouble(o);

	return	d;
}


/*
 * Functions : Cast functions
 * ---------------------------
 * Some of the function used below are based on C/python api
 * with more tolerance to different precisions and complex type.
 *	Parameterss	:
 *		C object
 *	Returns     :
 *		o  : python object
 * reference of the used c/python api function
 * ---------------------------------------------------
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_FromDoubles
 * https://docs.python.org/3/c-api/float.html#c.PyFloat_FromDouble
 * https://docs.python.org/3/c-api/long.html#c.PyLong_FromLongLong
 */

PyObject	*Complex_to_PyComplex(double complex c)
{
	double		real_part;
	double		imag_part;
	PyObject	*o;

	real_part = creal(c);
	imag_part = cimag(c);
	o = PyComplex_FromDoubles(real_part, imag_part);

	return o;
}

PyObject	*Bool_to_PyBool(bool b)
{
	return b == true ? Py_True : Py_False;
}

PyObject	*Int_to_PyLong(int64_t i)
{
	PyObject	*o;

	o = PyLong_FromLongLong((long long) i);

	return o;
}

PyObject	*Double_to_PyDouble(double d)
{
	PyObject	*o;

	o = PyFloat_FromDouble(d);

	return o;
}

/*
 * Function: _check_pyarray_dtype
 * --------------------
 * Check Python Object DataType:
 *
 * 	Parameters	:
 *		a 	  : python array object
 *      dtype : desired data type enum
 * 	Returns		:
 *		return true if no error occurred otherwise it will return false
 *      and raise TypeError exception
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_TYPE
 */


static bool _check_pyarray_dtype(PyArrayObject *a, int dtype)
{
	int curent_dtype;

	if (dtype == NO_TYPE_CHECK)
		return true;

	current_dtype = PyArray_TYPE(a);
	if (current_dtype != dtype)
	{
		PyErr_Format(PyExc_TypeError,
			"argument dtype must be %s, not %s",
			dataTypes[dtype],
			dataTypes[curent_dtype]);
		return false;
	}

	return true;
}

/*
 * Function: _check_pyarray_rank
 * --------------------
 * Check Python Object Rank:
 *
 * 	Parameters	:
 *		a 	  : python array object
 *      rank  : desired rank
 * 	Returns		:
 *		return true if no error occurred otherwise it will return false
 *      and raise TypeError exception
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_NDIM
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
 * 	Parameters	:
 *		a 	  : python array object
 *      flag  : desired order
 * 	Returns		:
 *		return true if no error occurred otherwise it will return false
 *      and raise NotImplementedError exception
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_CHKFLAGS
 */

static bool _check_pyarray_order(PyArrayObject *a, int flag)
{
	char	order;

	if (flag == NO_ORDER_CHECK)
		return true;

	if (rank > 1)
	{
		if (!PyArray_CHKFLAGS(array, flags))
		{
			order = flags == NPY_ARRAY_C_CONTIGUOUS ? 'C' : 'F';
			PyErr_Format(PyExc_NotImplementedError,
				"argument does not have the expected ordering (%c)", order);
			return false;
		}
	}

	return true;
}

/*
 * Function: _check_pyarray_type
 * --------------------
 * Check if Python Object is ArrayType:
 *
 * 	Parameters	:
 *		a 	  : python array object
 *
 * 	Returns		:
 *		return true if no error occurred otherwise it will return false
 *      and raise TypeError exception
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Check
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
 * Function: _check_array
 * --------------------
 * Check Python Object (DataType, Rank, Order):
 *
 * 	Parameters	:
 *		a 	  : python array object
 *      dtype : desired data type enum
 *		rank  : desired rank
 *		flag  : desired order flag
 * 	Returns		:
 *		return true if no error occurred otherwise it will return false
 */

static bool	_check_array(PyArrayObject *a, int dtype, int rank, int flag)
{
	if(!__check_pyarray_dtype(pyarray, dtype)) return false;

	if(!__check_pyarray_order(pyarray, order)) return false;

	if(!__check_pyarray_rank(pyarray, rank)) return false;

	return true;
}

/*
 * Function: pyarray_to_ndarray
 * ----------------------------
 * A Cast function that convert numpy array variable into ndarray variable,
 * by copying its information and data to a new variable of type ndarray struct
 * and return this variable to be used inside c code.
 * 	Parameters	:
 *		o 	  : python array object
 *      array : c ndarray object
 *      dtype : desired data type enum
 *		rank  : desired rank
 *		flag  : desired order flag
 * 	Returns		:
 *		return true if no error occurred otherwise it will return false
 *
 * reference of the used c/numpy api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html
 */

bool	pyarray_to_ndarray(PyObject *o, t_ndarray *array, int dtype, int rank, int flag)
{
	PyArrayObject	*pyarray;

	// Importing the API
	if (PyArray_API == NULL) import_array();
	// Array type c
	if (!_check_pyarray_type(a)) return false;

	pyarray = (PyArrayObject *)a;

	if (!_check_array(pyarray, dtype, rank, flag)) return false;

	array->nd          = PyArray_NDIM(pyarray);
	array->raw_data    = PyArray_DATA(pyarray);
	array->type_size   = PyArray_ITEMSIZE(pyarray);
	array->type        = PyArray_TYPE(pyarray);
	array->length      = PyArray_SIZE(pyarray);
	array->buffer_size = PyArray_NBYTES(pyarray);

	array->shape       = (int64_t*)malloc(sizeof(int64_t) * array->nd);
	memcpy(array->shape, PyArray_SHAPE(pyarray), sizeof(int64_t) * array->nd);

	array->strides     = (int64_t*)malloc(sizeof(int64_t) * array->nd);
	memcpy(array->strides, PyArray_STRIDES(pyarray), sizeof(int64_t) * array->nd);

	array->is_view     = 1;

	return true;
}

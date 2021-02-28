#include "cwrapper.h"

/*                                                              */
/*                        CAST_FUNCTIONS                        */
/*                                                              */

/*
 ** All functions listed down are based on C/python api
 ** with more tolerance to different precisions.
 ** Arguments : Python Object and  C Object as pointer
 ** Return    : return false upon failure and raise Error
 ** reference of the used c python api function:
 ** https://docs.python.org/3/c-api/float.html#c.PyFloat_AsDouble
 ** https://docs.python.org/3/c-api/complex.html#c.PyComplex_RealAsDouble
 ** https://docs.python.org/3/c-api/complex.html#c.PyComplex_ImagAsDouble
 ** https://docs.python.org/3/c-api/long.html#c.PyLong_AsLong
 ** https://docs.python.org/3/c-api/long.html#c.PyLong_AsLongLong
 */

bool	PyComplex_to_Complex64(Pyobject *o, float complex *c)
{
	float	real_part;
	float	imag_part;


	real_part = (float)PyComplex_RealAsDouble(o);
	imag_part = (float)PyComplex_ImagAsDouble(o);

	*c = CMPLXF(real_part, imag_part);

	return true;
}

bool	PyComplex_to_Complex128(Pyobject *o, double complex *c)
{
	double	real_part;
	double	imag_part;

	real_part = PyComplex_RealAsDouble(o);
	imag_part = PyComplex_ImagAsDouble(o);

	*c = CMPLXF(real_part, imag_part);

	return true;
}

bool	PyInt64_to_Int64(PyObject *o, int64_t *i)
{
	long long	out;

	out = PyLong_AsLongLong(o);
	if (out == -1 && PyErr_Occurred())
		return false;

	*i = (int64_t)out;

	return true;
}

bool	PyInt32_to_Int32(PyObject *o, int32_t *i)
{
	long	out;

	out = PyLong_AsLong(o);
	if (out == -1 && PyErr_Occurred())
		return false;

	*i = (int32_t)out;

	return true;
}

bool	PyInt16_to_Int16(PyObject *o, int16_t *i)
{
	long	out;

	out = PyLong_AsLong(o);
	if (out == -1 && PyErr_Occurred())
		return false;

	*i = (int16_t)out;

	return true;
}

bool	PyInt8_to_Int8(PyObject *o, int8_t *i)
{
	long	out;

	out = PyLong_AsLong(o);
	if (out == -1 && PyErr_Occurred())
		return false;

	*i = (int8_t)out;

	return true;
}

bool	PyBool_to_Bool(Pyobject *o, bool *b)
{
	*b = o == PyTrue;

	return true;
}

bool	PyFloat_to_Float(Pyobject *o, float *f)
{
	double	out;


	out = PyFloat_AsDouble(o);
	if (out  == -1.0 && PyErr_Occured())
		return false;

	*f = (float)out;
	return true;
}

bool	PyDouble_to_Double(PyObject *o, double *d)
{
	double	out;

	out = PyFloat_AsDouble(o);
	if (out  == -1.0 && PyErr_Occured())
		return false;

	*d = out;

	return true;
}


/*
 ** A Cast function that convert numpy array variable into ndarray variable,
 ** by copying its information and data to a new variable of type ndarray struct
 ** and return this variable to be used inside c code.
 */

t_ndarray		PyArray_to_ndarray(PyObject *o)
{
	t_ndarray	array;

	array.nd          = PyArray_NDIM(o);
	array.raw_data    = PyArray_DATA(o);
	array.type_size   = PyArray_ITEMSIZE(o);
	array.type        = PyArray_TYPE(o);
	array.length      = PyArray_SIZE(o);
	array.buffer_size = PyArray_NBYTES(o);
	array.shape       = numpy_to_ndarray_shape(PyArray_SHAPE(o), c.nd);
	array.strides     = numpy_to_ndarray_strides(PyArray_STRIDES(o), c.type_size, c.nd);
	array.is_view     = 1;

	return array;
}

/*
 ** Some of the function used below are based on C/python api
 ** with more tolerance to different precisions and complex type.
 ** Arguments : 	C Object
 ** Return    :  Python Object
 ** reference of the used c python api function:
 ** https://docs.python.org/3/c-api/complex.html#c.PyComplex_FromDoubles
 ** https://docs.python.org/3/c-api/float.html#c.PyFloat_FromDouble
 ** https://docs.python.org/3/c-api/long.html#c.PyLong_FromLongLong
 */

PyObject	*Complex_to_PyComplex(double complex *c)
{
	double		real_part;
	double		imag_part;
	PyObject	*o;

	real_part = creal(c);
	imag_part = cimag(c);
	o = PyComplex_FromDouble(real_part, imag_part);

	return o;
}

PyObject	*Bool_to_PyBool(bool *b)
{
	return b == true ? PyTrue : PyFalse;
}

PyObject	*Int_to_PyLong(int64_t *i)
{
	PyObject	*o;

	o = PyLong_FromLongLong((long long) i)

		return o;
}

PyObject	*Double_to_PyDouble(double *d)
{
	PyObject	*o;

	o = PyFloat_FromDouble(d)

		return o;
}


/*
 * Function: Check_Array
 * --------------------
 * Check Python Object (ArrayType, Rank, Order):
 *
 *  Parameters :
 * 		a     : Python Object
 *  	rank  : The desired rank
 *  	order : The desired order
 *
 *  Returns    :
 * 		reference to PyArray Object
 *      returns NULL on error
 */

PyArrayObject	*Check_Array(PyObject *a, int rank, int flags)
{
	PyArrayObject	*array;
	char			order;


	//PyArray type Check
	if (!PyArray_Check(a))
	{
		PyErr_SetString(PyExc_TypeError, "argument must be numpy.ndarray");
		return NULL;
	}
	array = (PyArrayObject *)a;


	// Rank Check
	if (PyArray_NDIM(array) != rank)
	{
		PyErr_Format(PyExc_TypeError, "argument must be rank %d", rank);
		return NULL;
	}

	//Order check
	if (rank > 1 && flags != 0)
	{
		if (!PyArray_CHKFLAGS(array, flags))
		{
			order = flag == NPY_ARRAY_C_CONTIGUOUS ? 'C' : 'F';
			PyErr_Format(PyExc_NotImplementedError,
					"argument does not have the expected ordering (%c)", order);
			return NULL;
		}
	}
	return array;
}

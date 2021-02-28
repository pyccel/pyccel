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

float complex PyComplex_to_Complex64(Pyobject *o)
{
	float complex	c;
	float			real_part;
	float			imag_part;


	real_part = (float)PyComplex_RealAsDouble(o);
	imag_part = (float)PyComplex_ImagAsDouble(o);

	c = CMPLXF(real_part, imag_part);

	return	c;
}

double complex	PyComplex_to_Complex128(Pyobject *o)
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

bool	PyBool_to_Bool(Pyobject *o)
{
	bool	b;

	b = o == PyTrue;

	return	b;
}

bool	PyFloat_to_Float(Pyobject *o)
{
	float	f;
	double	out;

	out = PyFloat_AsDouble(o);

	f = (float)out;

	return	f;
}

bool	PyDouble_to_Double(PyObject *o)
{
	double	d;

	d = PyFloat_AsDouble(o);

	return	d;
}


/*
 * Function: PyArray_to_ndarray
 * ----------------------------
 * A Cast function that convert numpy array variable into ndarray variable,
 * by copying its information and data to a new variable of type ndarray struct
 * and return this variable to be used inside c code.
 * 	Parameters	:
 *		o 	  : python array object
 * 	Returns		:
 *		array : c ndarray
 *
 * reference of the used c/numpy api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html
 */


t_ndarray		PyArray_to_ndarray(PyArrayObject *o)
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
 *               0 if the check order is not needed
 *
 *  Returns    :
 * 		reference to PyArray Object
 *      returns NULL on error
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html
 */

PyArrayObject	*Check_Array(PyObject *a, int rank, int flags)
{
	PyArrayObject	*array;
	char			order;

	//PyArray type Check
	if (!PyArray_Check(a))
	{
		PyErr_Format(PyExc_TypeError,
			"argument must be numpy.ndarray, not %s",
			 a == Py_None ? "None" : Py_TYPE(a)->tp_name);
		return NULL;
	}
	array = (PyArrayObject *)a;


	// Rank Check
	if (PyArray_NDIM(array) != rank)
	{
		PyErr_Format(PyExc_TypeError, "argument rank must be %d", rank);
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

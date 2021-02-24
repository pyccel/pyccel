#include "cwrapper.h"

/*                                                              */
/*                        CAST_FUNCTIONS                        */
/*                                                              */

// Python to C

int	PyComplex_to_Complex64(Pyobject *o, float complex *c)
{
	float	real_part;
	float	imag_part;

	if (PyArray_IsScalar(o, Complex64))
	{
		PyArray_ScalarAsCtype(o, c);
	}

	else if (PyComplex_Check(o))
	{
		real_part = (float)PyComplex_Real_AsDouble(o);
		imag_part = (float)PyComplex_Imag_AsDouble(o);

		*c = CMPLXF(real_part, imag_part);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "argument must be 64 bit complex");
		return 0;
	}
	return 1;
}

int	PyComplex_to_Complex128(Pyobject *o, double complex *c)
{
	double	real_part;
	double	imag_part;

	if (PyArray_Is_Scalar(o, Complex128))
	{
		PyArray_ScalarAsCtype(o, c);
	}
	else if (PyComplex_Check(o))
	{
		real_part = PyComplex_Real_AsDouble(o);
		imag_part = PyComplex_Imag_AsDouble(o);

		*c = CMPLX(real_part, imag_part);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "argument must be 128 bit complex");
		return 0;
	}
	return 1;
}

int	PyInt64_to_Int64(PyObject *o, int64_t *i)
{
	if (PyArray_Is_Scalar(o, Int64))
	{
		PyArray_ScalarAsCtype(o, i);
	}
	else if (PyLongCheck(o))
	{
		*i = (int64_t)PyLong_AsLongLong(o);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "argument must be 64 bit integer");
		return 0;
	}
	return 1;
}

int	PyInt32_to_Int32(PyObject *o, int32_t *i)
{
	if (PyArray_Is_Scalar(o, Int32))
	{
		PyArray_ScalarAsCtype(o, i);
	}
	else if (PyLongCheck(o))
	{
		*i = (int32_t)PyLong_AsLong(o);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "argument must be 32 bit integer");
		return 0;
	}
	return 1;
}

int	PyInt16_to_Int16(PyObject *o, int16_t *i)
{
	if (PyArray_Is_Scalar(o, Int16))
	{
		PyArray_ScalarAsCtype(o, i);
	}
	else if (PyLongCheck(o))
	{
		*i = (int16_t)PyLong_AsLong(o);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "argument must be 16 bit integer");
		return 0;
	}
	return 1;
}

int	PyInt8_to_Int8(PyObject *o, int8_t *i)
{
	if (PyArray_Is_Scalar(o, Int16))
	{
		PyArray_ScalarAsCtype(o, i);
	}
	else if (PyLongCheck(o))
	{
		*i = (int8_t)PyLong_AsLong(o);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "argument must be 8 bit integer");
		return 0;
	}
	return 1;
}


int	PyBool_to_Bool(Pyobject *o, bool *b)
{
	if (PyArray_Is_Scalar(o, Bool))
	{
		PyArray_ScalarAsCtype(o, b);
	}
	else if (PyBool_Check(o))
	{
		*b = o == PyTrue;
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "argument must be a boolean");
		return 0;
	}
	return 1;
}

int PyFloat_to_Float(Pyobject *o, float *f)
{
	if (PyArray_Is_Scalar(o, Float32))
	{
		PyArray_ScalarAsCtype(o, b);
	{
	else if (PyFloat_Check(o))
	{
		*f = (float)PyFloat_AsDouble(o);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "argument must be 32 bit float");
		return 0;
	}
	return 1;
}

int PyDouble_to_Double(PyObject *o, double *d)
{
	if (PyArray_Is_Scalar(o, Float64))
	{
		PyArray_ScalarAsCtype(o, b);
	}
	else if (PyFloat_Check(o))
	{
		*f = (float)PyFloat_AsDouble(o);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "argument must be 64 bit float");
		return 0;
	}
	return 1;
}

t_ndarray		PyArray_to_ndarray(PyObject *o)
{
	t_ndarray	array;

	c.nd          = PyArray_NDIM(o);
	c.raw_data    = PyArray_DATA(o);
	c.type_size   = PyArray_ITEMSIZE(o);
	c.type        = PyArray_TYPE(o);
	c.length      = PyArray_SIZE(o);
	c.buffer_size = PyArray_NBYTES(o);
	c.shape       = numpy_to_ndarray_shape(PyArray_SHAPE(o), c.nd);
	c.strides     = numpy_to_ndarray_strides(PyArray_STRIDES(o), c.type_size, c.nd);
	c.is_view     = 1;

	return array;
}

// C to Python

PyObject	*Complex64_to_PyComplex(float complex c)
{
	float		real_part;
	float		imag_part;
	PyObject	*o;

	real_part = creal(c);
	imag_part = cimag(c);
	o = PyComplex_FromDouble((double)real_part, (double)imag_part);

	return o;
}

PyObject	*Complex64_to_PyComplex(double complex c)
{
	double		real_part;
	double		imag_part;
	PyObject	*o;

	real_part = creal(c);
	imag_part = cimag(c);
	o = PyComplex_FromDouble(real_part, imag_part);

	return o;
}

PyObject	*Bool_to_PyBool(bool b)
{
	PyObject	*o;

	return b == true ? PyTrue : PyFalse;
}

/*  CHECK FUNCTION  */
bool	PyArray_Check_Rank(PyArrayObject *a, int rank)
{
}

bool	PyArray_Check_Type(PyArrayObject *a, int rank)
{

}


bool	PyArray_Check(PyArrayObject *a, int rank, int dtype)
{

}
{
	char	*error;

	if (PyArray_NDIM(a) != rank)
	{
		PyErr_Format(PyExc_TypeError, "Arguments rank must be %d", rank);
		return 0;
	}
	else if(PyArray_TYPE(a) != dtype)
	{
		PyErr_Format(PyExc_TypeError, "Arguments type must be %d", dtype);
		return 0;
	}
}

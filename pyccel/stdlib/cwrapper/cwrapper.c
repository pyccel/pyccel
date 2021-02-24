#include "cwrapper.h"

/*                                                              */
/*                        CAST_FUNCTIONS                        */
/*                                                              */

// Python to C

float complex	PyComplex_to_Complex64(Pyobject *o)
{
	float			real_part;
	float			imag_part;
	float complex	c;

	real_part = PyComplex_Real_AsDouble(o);
	imag_part = PyComplex_Imag_AsDouble(o);

	c = CMPLXF(real_part, imag_part);
	return c;
}

double complex	PyComplex_to_Complex128(Pyobject *o)
{
	double			real_part;
	double			imag_part;
	double complex	c;

	real_part = PyComplex_Real_AsDouble(o);
	imag_part = PyComplex_Imag_AsDouble(o);

	c = CMPLX(real_part, imag_part);
	return c;
}

bool			PyBool_to_Bool(Pyobject *o)
{
	bool	b;

	b = o == PyTrue;

	return b;
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

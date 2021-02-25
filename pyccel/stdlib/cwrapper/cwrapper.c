#include "cwrapper.h"

/*                                                              */
/*                        CAST_FUNCTIONS                        */
/*                                                              */

// Python to C


const bool	g_t_py_to_c[9] =
{
	&PyBool_to_Bool,
	&PyInt8_to_Int8,
	&PyInt16_to_Int16,
	&PyInt32_to_Int32,
	&PyInt64_to_Int64,

	&PyFloat_to_Float,
	&PyDouble_to_Doube,

	&PyComplex_to_Complex64,
	&PyComplex_to_Complex128,
};



static int	PyComplex_to_Complex64(Pyobject *o, float complex *c)
{
	float	real_part;
	float	imag_part;


	real_part = (float)PyComplex_RealAsDouble(o);
	imag_part = (float)PyComplex_ImagAsDouble(o);

	*c = CMPLXF(real_part, imag_part);

	return true;
}

static int	PyComplex_to_Complex128(Pyobject *o, double complex *c)
{
	double	real_part;
	double	imag_part;

	real_part = PyComplex_RealAsDouble(o);
	imag_part = PyComplex_ImagAsDouble(o);

	*c = CMPLXF(real_part, imag_part);

	return true;
}

static int	PyInt64_to_Int64(PyObject *o, int64_t *i)
{
	long long	out;

	out = PyLong_AsLongLong(o);
	if (out == -1 && PyErr_Occurred())
		return false;

	*i = (int64_t)out;

	return true;
}

static int	PyInt32_to_Int32(PyObject *o, int32_t *i)
{
	long	out;

	out = PyLong_AsLong(o);
	if (out == -1 && PyErr_Occurred())
		return false;

	*i = (int32_t)out;

	return true;
}

static bool	PyInt16_to_Int16(PyObject *o, int16_t *i)
{
	long	out;

	out = PyLong_AsLong(o);
	if (out == -1 && PyErr_Occurred())
		return false;

	*i = (int16_t)out;

	return true;
}

static bool	PyInt8_to_Int8(PyObject *o, int8_t *i)
{
	long	out;

	out = PyLong_AsLong(o);
	if (out == -1 && PyErr_Occurred())
		return false;

	*i = (int8_t)out;

	return true;
}

static bool	PyBool_to_Bool(Pyobject *o, bool *b)
{
	*b = o == PyTrue;

	return true;
}

static bool	PyFloat_to_Float(Pyobject *o, float *f)
{
	double	out;


	out = PyFloat_AsDouble(o);
	if (out  == -1.0 && PyErr_Occured())
		return false;

	*f = (float)out;
	return true;
}

static bool	PyDouble_to_Double(PyObject *o, double *d)
{
	double	out;

	out = PyFloat_AsDouble(o);
	if (out  == -1.0 && PyErr_Occured())
		return false;

	*d = out;

	return true;
}

bool		PyObject_As_Ctype(PyObject *o, void *v, int type)
{
	return g_t_py_to_c[type](o, v);
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

PyObject	*Int64_to_PyInt64(int64_t i)
{
	PyObject	*o;

	o = PyLong_FromLongLong(i)

		return 0;
}


/*  CHECK FUNCTION  */
bool	PyArray_Check_Rank(PyObject *a, int rank)
{
	return PyArray_NDIM(a) == rank;
}

bool	PyArray_Check_Type(PyArrayObject *a, int type)
{
	return PyArray_TYPE(a) == type;
}

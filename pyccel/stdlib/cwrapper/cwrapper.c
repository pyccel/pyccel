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

bool	PyArray_Check(PyArrayObject *a, int rank, int dtype)
{
	char	*error;

	if (PyArray_NDIM(a) != rank)
	{
		PyErr_SetString(PyExc_TypeError, "\"x_tmp must have rank \"");
		return 0;
	}
	else if(PyArray_TYPE(a) != dtype)
	{
		return 0;
	}
}

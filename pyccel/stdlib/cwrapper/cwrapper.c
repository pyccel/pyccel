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

	//https://numpy.org/doc/1.17/reference/c-api.array.html#c.PyArray_IsScalar
	//https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_ScalarAsCtype
	if (PyArray_IsScalar(o, Complex64))
		PyArray_ScalarAsCtype(o, &c);

	else
	{
		real_part = (float)PyComplex_RealAsDouble(o);
		imag_part = (float)PyComplex_ImagAsDouble(o);

		c = CMPLXF(real_part, imag_part);
	}
	return	c;
}
//-----------------------------------------------------//
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
//-----------------------------------------------------//
int64_t	PyInt64_to_Int64(PyObject *o)
{
	int64_t		i;
	long long	out;

	out = PyLong_AsLongLong(o);

	i = (int64_t)out;

	return	i;
}
//-----------------------------------------------------//
int32_t	PyInt32_to_Int32(PyObject *o)
{
	int32_t	i;
	long	out;

	out = PyLong_AsLong(o);

	i = (int32_t)out;

	return	i;
}
//-----------------------------------------------------//
int16_t	PyInt16_to_Int16(PyObject *o)
{
	int16_t	i;
	long	out;

	out = PyLong_AsLong(o);

	i = (int16_t)out;

	return	i;
}
//-----------------------------------------------------//
int8_t	PyInt8_to_Int8(PyObject *o)
{
	int8_t	i;
	long	out;

	out = PyLong_AsLong(o);

	i = (int8_t)out;

	return	i;
}
//-----------------------------------------------------//
bool	PyBool_to_Bool(PyObject *o)
{
	bool	b;

	b = o == Py_True;

	return	b;
}
//-----------------------------------------------------//
float	PyFloat_to_Float(PyObject *o)
{
	float	f;
	double	out;

	out = PyFloat_AsDouble(o);

	f = (float)out;

	return	f;
}
//-----------------------------------------------------//
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
 *		o	        : the python object
 *		hard_check  : boolean
 *			true if we need to check the exact precision otherwise false
 *	Returns     :
 *		boolean : logic statement responsible for checking python data type
 * reference of the used c/python api function
 * ---------------------------------------------------
 * https://docs.python.org/3/c-api/long.html#c.PyLong_Check
 * https://docs.python.org/3/c-api/complex.html#c.PyComplex_Check
 * https://docs.python.org/3/c-api/float.html#c.PyFloat_Check
 * https://docs.python.org/3/c-api/bool.html#c.PyBool_Check
 * https://numpy.org/doc/1.17/reference/c-api.array.html#c.PyArray_IsScalar
 */

PyObject	*Complex128_to_PyComplex(double complex *c)
{
	double		real_part;
	double		imag_part;
	PyObject	*o;

	real_part = creal(*c);
	imag_part = cimag(*c);
	o = PyComplex_FromDoubles(real_part, imag_part);

	return o;
}
//-----------------------------------------------------//
PyObject	*Complex64_to_PyComplex(float complex *c)
{
	float		real_part;
	float		imag_part;
	PyObject	*o;

	real_part = crealf(*c);
	imag_part = cimagf(*c);
	o = PyComplex_FromDoubles((double) real_part, (double) imag_part);

	return o;
}
//-----------------------------------------------------//
PyObject	*Bool_to_PyBool(bool *b)
{
	return *b == true ? Py_True : Py_False;
}
//-----------------------------------------------------//
PyObject	*Int64_to_PyLong(int64_t *i)
{
	PyObject	*o;

	o = PyLong_FromLongLong((long long) *i);

	return o;
}
//-----------------------------------------------------//
PyObject	*Int32_to_PyLong(int32_t *i)
{
	PyObject	*o;

	o = PyLong_FromLongLong((long long) *i);

	return o;
}
//-----------------------------------------------------//
PyObject	*Int16_to_PyLong(int16_t *i)
{
	PyObject	*o;

	o = PyLong_FromLongLong((long long) *i);

	return o;
}
//--------------------------------------------------------//
PyObject	*Int8_to_PyLong(int8_t *i)
{
	PyObject	*o;

	o = PyLong_FromLongLong((long long) *i);

	return o;
}
//--------------------------------------------------------//
PyObject	*Double_to_PyDouble(double *d)
{
	PyObject	*o;

	o = PyFloat_FromDouble(*d);

	return o;
}
//--------------------------------------------------------//
PyObject	*Float_to_PyDouble(float *d)
{
	PyObject	*o;

	o = PyFloat_FromDouble((double)*d);

	return o;
}


/*
 * Functions : Check type functions
 * ---------------------------
 * Some of the function used below are based on C/python api
 * and numpy/c api with more tolerance to different precisions,
 * different system architectures and complex type.
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
	#endif

	if (hard_check == true)
		return PyArray_IsScalar(o, Int32);

	return PyLong_Check(o) || PyArray_IsScalar(o, Int32);
}
//--------------------------------------------------------//
bool    PyIs_Int64(PyObject *o, bool hard_check)
{
	#ifndef _WIN32
		return PyLong_Check(o) || PyArray_IsScalar(o, Int64);
	#endif

	if (hard_check == true)
		return PyArray_IsScalar(o, Int64);

	return PyLong_Check(o) || PyArray_IsScalar(o, Int64);
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





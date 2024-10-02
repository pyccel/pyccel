/* -------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file  */
/* or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. */
/* -------------------------------------------------------------------------------------- */

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
 * https://numpy.org/doc/stable/reference/c-api/array.html?highlight=pyarray_scalar#c.PyArray_Scalar
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
PyObject	*Complex128_to_NumpyComplex(double complex *c)
{
    return PyArray_Scalar(c, PyArray_DescrFromType(NPY_COMPLEX128), NULL);
}
//-----------------------------------------------------//
PyObject	*Complex64_to_NumpyComplex(float complex *c)
{
    return PyArray_Scalar(c, PyArray_DescrFromType(NPY_COMPLEX64), NULL);
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
PyObject	*Int64_to_NumpyLong(int64_t *i)
{
    return PyArray_Scalar(i, PyArray_DescrFromType(NPY_INT64), NULL);
}
//-----------------------------------------------------//
PyObject	*Int32_to_NumpyLong(int32_t *i)
{
    return PyArray_Scalar(i, PyArray_DescrFromType(NPY_INT32), NULL);
}
//-----------------------------------------------------//
PyObject	*Int16_to_NumpyLong(int16_t *i)
{
    return PyArray_Scalar(i, PyArray_DescrFromType(NPY_INT16), NULL);
}
//--------------------------------------------------------//
PyObject	*Int8_to_NumpyLong(int8_t *i)
{
    return PyArray_Scalar(i, PyArray_DescrFromType(NPY_INT8), NULL);
}
//--------------------------------------------------------//
PyObject	*Double_to_PyDouble(double *d)
{
	return PyFloat_FromDouble(*d);
}
//--------------------------------------------------------//
PyObject	*Double_to_NumpyDouble(double *d)
{
    return PyArray_Scalar(d, PyArray_DescrFromType(NPY_DOUBLE), NULL);
}
//--------------------------------------------------------//
PyObject	*Float_to_NumpyDouble(float *d)
{
    return PyArray_Scalar(d, PyArray_DescrFromType(NPY_FLOAT), NULL);
}


/*
 * Functions : Numpy array handling functions
 */

void get_strides_and_shape_from_numpy_array(PyObject* arr, int64_t shape[], int64_t strides[])
{
    PyArrayObject* a = (PyArrayObject*)(arr);
    int nd = PyArray_NDIM(a);

    PyArrayObject* base = (PyArrayObject*)PyArray_BASE(a);

    if (base == NULL) {
        npy_intp* np_shape = PyArray_SHAPE(a);
        for (int i = 0; i < nd; ++i) {
            shape[i] = np_shape[i];
            strides[i] = 1;
        }
    }
    else {
        npy_intp current_stride = PyArray_ITEMSIZE(a);
        npy_intp* np_strides = PyArray_STRIDES(a);
        npy_intp* np_shape = PyArray_SHAPE(a);
        for (int i = 0; i < nd; ++i) {
            shape[i] = np_shape[i];
            strides[i] = np_strides[i] / current_stride;
            current_stride *= shape[i];
        }
    }
}

void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
}

#ifdef _WIN32
PyObject* to_pyarray(int nd, enum NPY_TYPES typenum, void* data, int32_t shape[], bool c_order, bool release_memory)
#else
PyObject* to_pyarray(int nd, enum NPY_TYPES typenum, void* data, int64_t shape[], bool c_order, bool release_memory)
#endif
{
    int FLAGS;
    if (nd == 1) {
        FLAGS = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE;
    }
    else if (c_order) {
        FLAGS = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE;
    }
    else {
        FLAGS = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_WRITEABLE;
    }

    npy_intp npy_shape[nd];

    for (int i=0; i<nd; ++i) {
        npy_shape[i] = shape[i];
    }

    PyObject* arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(typenum),
                                         nd, npy_shape, NULL, data, FLAGS, NULL);
    if (release_memory) {
        // Add a capsule base to ensure that memory is freed.
        PyObject* base = PyCapsule_New(data, NULL, capsule_cleanup);
        PyArray_SetBaseObject((PyArrayObject*)arr, base);
    }
    return arr;
}


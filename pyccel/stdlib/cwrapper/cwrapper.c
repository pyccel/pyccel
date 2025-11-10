/* -------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file  */
/* or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. */
/* -------------------------------------------------------------------------------------- */

#include "cwrapper.h"




const int NO_TYPE_CHECK = -1;
const int NO_ORDER_CHECK = -1;



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
    PyObject* result = (*b) ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
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

/**
 * Calculate the shapes and strides necessary to pass an array to low-level code.
 *
 * If an array is a view on another array then it is passed to low-level code by
 * saving it into a view with the correct dimensionality and slicing that object.
 * This function returns the 3 objects necessary for this operation.
 * Namely:
 *   - base_shape : The shape of the view with the correct dimensionality.
 *   - ubounds : The upper bounds of the slice.
 *   - strides : The strides of the slice.
 *
 * E.g.
 * ```python
 * a = np.ones((2,3,4,5))
 * b = a[:,2,::2,:]
 * ```
 * Here b has dimensionality 3 so the base_shape, ubounds and strides each contain
 * 3 elements:
 *   - base_shape : [4,4,5]
 *   - ubound : [4,4,5]
 *   - strides : [3,2,1]
 * The lbound is handled with offsets. This calculation is already carried out by
 * NumPy. The data pointer is provided with the offset.
 *
 * @param[in] arr The array we wish to pass.
 * @param[out] base_shape The shape of the view before slicing.
 * @param[out] ubounds The upper bounds of the slice.
 * @param[out] strides The strides of the slice.
 * @param[in] c_order True if the array has C-ordering, False otherwise.
 */
void get_strides_and_shape_from_numpy_array(PyObject* arr, int64_t base_shape[], int64_t ubounds[], int64_t strides[], bool c_order)
{
    // Get information about the array
    PyArrayObject* a = (PyArrayObject*)(arr);
    int nd = PyArray_NDIM(a);

    // Determine whether the array is a sub-view of a different array
    PyArrayObject* base = (PyArrayObject*)PyArray_BASE(a);
    if (base == NULL) {
        npy_intp* np_shape = PyArray_SHAPE(a);
        for (int i = 0; i < nd; ++i) {
            base_shape[i] = np_shape[i];
            ubounds[i] = np_shape[i];
            strides[i] = 1;
        }
    }
    else {
        // Calculate base_shape, ubounds, strides
        npy_intp itemsize = PyArray_ITEMSIZE(a);
        npy_intp* np_strides = PyArray_STRIDES(a);
        npy_intp* np_shape = PyArray_SHAPE(a);
        int64_t current_shape = itemsize;
        if (c_order) {
            // If code is C-ordered then the smallest stride is the last element
            strides[nd-1] = np_strides[nd-1] / itemsize;
            for (int i = nd-1; i >= 1; --i) {
                base_shape[i] = np_strides[i-1] / current_shape;
                current_shape *= base_shape[i];
                strides[i-1] = np_strides[i-1] / current_shape;
            }
            base_shape[0] = np_shape[0] * strides[0];
        }
        else {
            // If code is F-ordered then the smallest stride is the first element
            strides[0] = np_strides[0] / itemsize;
            for (int i = 0; i < nd-1; ++i) {
                base_shape[i] = np_strides[i+1] / current_shape;
                current_shape *= base_shape[i];
                strides[i+1] = np_strides[i+1] / current_shape;
            }
            base_shape[nd-1] = np_shape[nd-1] * strides[nd-1];
        }
        for (int i = 0; i < nd; ++i) {
            ubounds[i] = np_shape[i] * strides[i];
        }
    }
}

void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
}

PyObject* to_pyarray(int nd, enum NPY_TYPES typenum, void* data, int32_t shape[], bool c_order, bool release_memory)
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

/*
 * Function: _check_pyarray_dtype
 * --------------------
 * Check Python Object DataType:
 *
 * 	Parameters	:
 *		a 	  : python array object
 *      dtype : desired data type enum
 * 	Returns		:
 *		return NULL if no error occurred otherwise it will return the
 *      message to be reported in a TypeError exception
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_TYPE
 */
static char*	_check_pyarray_dtype(PyArrayObject *a, int dtype)
{
	int current_dtype;

	if (dtype == NO_TYPE_CHECK)
		return NULL;

	current_dtype = PyArray_TYPE(a);
	if (current_dtype != dtype)
	{
        PyObject* current_type_name = PyObject_Str((PyObject*)PyArray_DESCR(a)->typeobj);
        PyObject* expected_type_name = PyObject_Str(PyArray_TypeObjectFromType(dtype));
        Py_ssize_t c_size;
        const char* current_name = PyUnicode_AsUTF8AndSize(current_type_name, &c_size);
        const char* expected_name = PyUnicode_AsUTF8AndSize(expected_type_name, &c_size);
        char* error = (char *)malloc(200);
        sprintf(error, "argument dtype must be %s, not %s",
			expected_name,
			current_name);
		return error;
	}

	return NULL;
}

/*
 * Function: _check_pyarray_rank
 * --------------------
 * Check Python Object Rank:
 *
 * 	Parameters	:
 *		a 	  : python array object
 *      rank  : desired rank
 *      allow_empty : Indicate if the array can be empty (empty arrays raise an error in STC).
 * 	Returns		:
 *		return NULL if no error occurred otherwise it will return the
 *      message to be reported in a TypeError exception
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_NDIM
 */
static char* _check_pyarray_rank(PyArrayObject *a, int rank, bool allow_empty)
{
    int current_rank;

    current_rank = PyArray_NDIM(a);
    if (current_rank != rank)
    {
        char* error = (char *)malloc(200);
        sprintf(error, "argument rank must be %d, not %d",
            rank,
            current_rank);
        return error;
    }

    if (!allow_empty) {
        npy_intp* np_shape = PyArray_SHAPE(a);
        for (int i = 0; i < rank; ++i) {
            if (np_shape[i] == 0) {
                char* error = (char *)malloc(200);
                sprintf(error, "Array has size 0 in dimension %d", i);
                return error;
            }
        }
    }

    return NULL;
}

/*
 * Function: _check_pyarray_order
 * --------------------
 * Check Python Object Order:
 *
 * 	Parameters	:
 *		a 	  : python array object
 *      flag  : A flag that is recognised by NumPy's PyArray_CHKFLAGS function.
 * 	Returns		:
 *		return NULL if no error occurred otherwise it will return the
 *      message to be reported in a TypeError exception
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_CHKFLAGS
 */
static char* _check_pyarray_order(PyArrayObject *a, int flag)
{
    if (flag == NO_ORDER_CHECK)
        return NULL;

    bool valid = true;
    if (flag == NPY_ARRAY_C_CONTIGUOUS) {
        int nd = PyArray_NDIM(a);
        npy_intp* np_strides = PyArray_STRIDES(a);
        for (int i = 1; i<nd; ++i) {
            valid = valid & (np_strides[i-1] >= np_strides[i]);
        }
    }
    else if (flag == NPY_ARRAY_F_CONTIGUOUS) {
        int nd = PyArray_NDIM(a);
        npy_intp* np_strides = PyArray_STRIDES(a);
        for (int i = 1; i<nd; ++i) {
            valid = valid & (np_strides[i-1] <= np_strides[i]);
        }
    }
    else {
        valid = PyArray_CHKFLAGS(a, flag);
    }

    if (!valid)
    {
        char order = (flag == NPY_ARRAY_C_CONTIGUOUS ? 'C' : (flag == NPY_ARRAY_F_CONTIGUOUS ? 'F' : '?'));
        char* error = (char *)malloc(200);
        sprintf(error, "argument does not have the expected ordering (%c)", order);
        return error;
    }

    return NULL;
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
 *		return NULL if no error occurred otherwise it will return the
 *      message to be reported in a TypeError exception
 * reference of the used c/python api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Check
 */
static char* _check_pyarray_type(PyObject *a)
{
	if (!PyArray_Check(a))
	{
        char* error = (char *)malloc(200);
        sprintf(error, "argument must be numpy.ndarray, not %s",
			 a == Py_None ? "None" : Py_TYPE(a)->tp_name);
        return error;
	}

	return NULL;
}

/*
 * Function: pyarray_check
 * --------------------
 * Check Python Object (DataType, Rank, Order):
 *
 * 	Parameters	:
 * 	    name  : the name of the argument (used for error output)
 *		a 	  : python array object
 *      dtype : desired data type enum
 *		rank  : desired rank
 *		flag  : desired order flag
 *      allow_empty : Indicate if the array can be empty (empty arrays raise an error in STC).
 * 	Returns		:
 *		return true if no error occurred otherwise it will return false
 */
bool	pyarray_check(const char* name, PyObject *o, int dtype, int rank, int flag, bool allow_empty)
{
    char* array_type = _check_pyarray_type(o);
	if (array_type != NULL) {
		PyErr_Format(PyExc_TypeError, array_type);
        free(array_type);
        return false;
    }

    PyArrayObject* a = (PyArrayObject*)o;

    bool correct_type = true;
    char error[800];
    sprintf(error, "Wrong argument type for argument %s : ", name);

	// check array element type / rank / order
    char* array_dtype = _check_pyarray_dtype(a, dtype);
    if (array_dtype != NULL) {
        strcat(error, array_dtype);
        free(array_dtype);
        correct_type = false;
    }

    char* array_rank = _check_pyarray_rank(a, rank, allow_empty);
    if (array_rank != NULL) {
        if (!correct_type)
            strcat(error, ", ");
        strcat(error, array_rank);
        free(array_rank);
        correct_type = false;
    }

    if (rank > 1) {
        char* array_order = _check_pyarray_order(a, flag);
        if (array_order != NULL) {
            if (!correct_type)
                strcat(error, ", ");
            strcat(error, array_order);
            free(array_order);
            correct_type = false;
        }
    }

    if (!correct_type) {
		PyErr_SetString(PyExc_TypeError, error);
    }
    return correct_type;
}

bool	is_numpy_array(PyObject *o, int dtype, int rank, int flag, bool allow_empty)
{
    char* array_type = _check_pyarray_type(o);
	if (array_type != NULL) {
        free(array_type);
        return false;
    }

    PyArrayObject* a = (PyArrayObject*)o;

	// check array element type / rank / order
    char* array_dtype = _check_pyarray_dtype(a, dtype);
	if(array_dtype != NULL) {
        free(array_dtype);
        return false;
    }

    char* array_rank = _check_pyarray_rank(a, rank, allow_empty);
	if(array_rank != NULL) {
        free(array_rank);
        return false;
    }

    if (rank > 1) {
        char* array_order = _check_pyarray_order(a, flag);
        if(array_order != NULL) {
            free(array_order);
            return false;
        }
    }

	return true;
}

extern inline int64_t	PyInt64_to_Int64(PyObject *object);
extern inline int32_t	PyInt32_to_Int32(PyObject *object);
extern inline int16_t	PyInt16_to_Int16(PyObject *object);
extern inline int8_t	PyInt8_to_Int8(PyObject *object);
extern inline bool	PyBool_to_Bool(PyObject *object);
extern inline float	PyFloat_to_Float(PyObject *object);
extern inline double	PyDouble_to_Double(PyObject *object);
extern inline bool    PyIs_NativeInt(PyObject *o);
extern inline bool    PyIs_Int8(PyObject *o);
extern inline bool    PyIs_Int16(PyObject *o);
extern inline bool    PyIs_Int32(PyObject *o);
extern inline bool    PyIs_Int64(PyObject *o);
extern inline bool    PyIs_NativeFloat(PyObject *o);
extern inline bool    PyIs_Float(PyObject *o);
extern inline bool    PyIs_Double(PyObject *o);
extern inline bool    PyIs_Bool(PyObject *o);
extern inline bool    PyIs_NativeComplex(PyObject *o);
extern inline bool    PyIs_Complex128(PyObject *o);
extern inline bool    PyIs_Complex64(PyObject *o);

/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#include "cwrapper_ndarrays.h"

/*
 * Function : _numpy_to_ndarray_strides
 * --------------------
 * Convert numpy strides to nd_array strides, and return it in a new array, to
 * avoid the problem of different implementations of strides in numpy and ndarray.
 * Parameters :
 *     np_strides : npy_intp array
 *     type_size  : data type enum
 *     nd : size of the array
 *
 * Returns    :
 *     ndarray_strides : a new array with new strides values
 */
static int64_t	*_numpy_to_ndarray_strides(npy_intp  *np_strides, int type_size, int nd)
{
    int64_t *ndarray_strides;

    ndarray_strides = (int64_t*)malloc(sizeof(int64_t) * nd);
    for (int i = 0; i < nd; i++)
        ndarray_strides[i] = (int64_t) np_strides[i] / type_size;

    return ndarray_strides;
}

static npy_intp	*_ndarray_to_numpy_strides(int64_t  *nd_strides, int32_t type_size, int nd)
{
    npy_intp *numpy_strides;

    numpy_strides = (npy_intp*)malloc(sizeof(npy_intp) * nd);
    for (int i = 0; i < nd; i++)
        numpy_strides[i] = (npy_intp) nd_strides[i] * type_size;

    return numpy_strides;
}


/*
 * Function : _numpy_to_ndarray_shape
 * --------------------
 * Copy numpy shape to nd_array shape, and return it in a new array, to
 * avoid the problem of variation of system architecture because numpy shape
 * is not saved in fixed length type.
 * Parameters :
 *     np_shape : npy_intp array
 *     nd : size of the array
 *
 * Returns    :
 *     ndarray_strides : new array
*/
static int64_t     *_numpy_to_ndarray_shape(npy_intp  *np_shape, int nd)
{
    int64_t *nd_shape;

    nd_shape = (int64_t*)malloc(sizeof(int64_t) * nd);
    for (int i = 0; i < nd; i++)
        nd_shape[i] = (int64_t) np_shape[i];
    return nd_shape;
}

static npy_intp *_ndarray_to_numpy_shape(int64_t *nd_shape, int nd)
{
    npy_intp *np_shape;

    np_shape = (npy_intp*)malloc(sizeof(npy_intp) * nd);
    for (int i = 0; i < nd; i++)
        np_shape[i] = (npy_intp) nd_shape[i];
    return np_shape;
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
bool	check_pyarray_dtype(PyArrayObject *a, int dtype)
{
	int current_dtype;

	if (dtype == NO_TYPE_CHECK)
		return true;

	current_dtype = PyArray_TYPE(a);
	if (current_dtype != dtype)
	{
		PyErr_Format(PyExc_TypeError,
			"argument dtype must be %s, not %s",
			dataTypes[dtype],
			dataTypes[current_dtype]);
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

	if (flag == NO_ORDER_CHECK)
		return true;

	if (!PyArray_CHKFLAGS(a, flag))
	{
		char order = (flag == NPY_ARRAY_C_CONTIGUOUS ? 'C' : (flag == NPY_ARRAY_F_CONTIGUOUS ? 'F' : '?'));
		PyErr_Format(PyExc_NotImplementedError,
			"argument does not have the expected ordering (%c)", order);
		return false;
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

enum NPY_TYPES get_numpy_type(t_ndarray *o)
{
    enum e_types nd_type = o->type;
    enum NPY_TYPES npy_type;
    switch (nd_type)
    {
        case nd_bool:
            npy_type = NPY_BOOL;
            break;
        case nd_int8:
            npy_type = NPY_INT8;
            break;
        case nd_int16:
            npy_type = NPY_INT16;
            break;
        case nd_int32:
            npy_type = NPY_INT32;
            break;
        case nd_int64:
            npy_type = NPY_INT64;
            break;
        case nd_float:
            npy_type = NPY_FLOAT;
            break;
        case nd_double:
            npy_type = NPY_DOUBLE;
            break;
        case nd_cfloat:
            npy_type = NPY_CFLOAT;
            break;
        case nd_cdouble:
            npy_type = NPY_CDOUBLE;
            break;
        default:
            printf("Unknown data type\n");
            exit(1);
    }
    return npy_type;
}

enum e_types get_ndarray_type(PyArrayObject *a)
{
    enum NPY_TYPES npy_type = PyArray_TYPE(a);
    enum e_types nd_type;
    switch (npy_type)
    {
        case NPY_BOOL:
            nd_type = nd_bool;
            break;
        case NPY_INT8:
            nd_type = nd_int8;
            break;
        case NPY_INT16:
            nd_type = nd_int16;
            break;
        case NPY_INT32:
            nd_type = nd_int32;
            break;
        case NPY_INT64:
            nd_type = nd_int64;
            break;
        case NPY_FLOAT:
            nd_type = nd_float;
            break;
        case NPY_DOUBLE:
            nd_type = nd_double;
            break;
        case NPY_CFLOAT:
            nd_type = nd_cfloat;
            break;
        case NPY_CDOUBLE:
            nd_type = nd_cdouble;
            break;
        default:
            printf("Unknown data type\n");
            exit(1);
    }
    return nd_type;
}

/* converting numpy array to c nd array*/
t_ndarray	pyarray_to_ndarray(PyArrayObject *a)
{
	t_ndarray		array;

	array.nd          = PyArray_NDIM(a);
	array.raw_data    = PyArray_DATA(a);
	array.type_size   = PyArray_ITEMSIZE(a);
	array.type        = get_ndarray_type(a);
	array.length      = PyArray_SIZE(a);
	array.buffer_size = PyArray_NBYTES(a);
	array.shape       = _numpy_to_ndarray_shape(PyArray_SHAPE(a), array.nd);
	array.strides     = _numpy_to_ndarray_strides(PyArray_STRIDES(a), array.type_size, array.nd);

	array.is_view     = 1;

	return array;
}

PyObject* ndarray_to_pyarray(t_ndarray *o)
{
    int FLAGS;
    if (o->nd == 1) {
        FLAGS = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE;
    }
    else {
        FLAGS = 0;
    }

    enum NPY_TYPES npy_type = get_numpy_type(o);

    return PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(npy_type),
            o->nd, _ndarray_to_numpy_shape(o->shape, o->nd),
            _ndarray_to_numpy_strides(o->strides, o->type_size, o->nd),
            o->raw_data, FLAGS, NULL);
}

PyObject* c_ndarray_to_pyarray(t_ndarray *o)
{
    int FLAGS = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE;

    enum NPY_TYPES npy_type = get_numpy_type(o);

    return PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(npy_type),
            o->nd, _ndarray_to_numpy_shape(o->shape, o->nd),
            _ndarray_to_numpy_strides(o->strides, o->type_size, o->nd),
            o->raw_data, FLAGS, NULL);
}

PyObject* fortran_ndarray_to_pyarray(t_ndarray *o)
{
    int FLAGS = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_WRITEABLE;
    return PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(o->type),
            o->nd, _ndarray_to_numpy_shape(o->shape, o->nd),
            _ndarray_to_numpy_strides(o->strides, o->type_size, o->nd),
            o->raw_data, FLAGS, NULL);
}

/*
 * Function: pyarray_check
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
bool	pyarray_check(PyArrayObject *o, int dtype, int rank, int flag)
{
	if (!_check_pyarray_type((PyObject *)o)) return false;

	// check array element type / rank / order
	if(!check_pyarray_dtype(o, dtype)) return false;

	if(!_check_pyarray_rank(o, rank)) return false;

	if(rank > 1 && !_check_pyarray_order(o, flag)) return false;

	return true;
}


/*
 * Function: nd_ndim
 * --------------------
 * Return the shape in the n dimension.
 *
 * 	Parameters	:
 *		a 	  : python array object
 *      index : dimension index
 * 	Returns		:
 *		return 0 if object is NULL or shape at indexed dimension
 * reference of the used c/numpy api function
 * -------------------------------------------
 * https://numpy.org/doc/1.17/reference/c-api.array.html#c.PyArray_DIM
 */
int     nd_ndim(t_ndarray *a, int n)
{
	if (a == NULL)
		return 0;

	return a->shape[n];
}


/*
 * Function: nd_data
 * --------------------
 * Return data pointed by array
 *
 * 	Parameters	:
 *		a 	  : python array object
 * 	Returns		:
 *		return NULL if object is NULL or the data of the array
 * reference of the used c/numpy api function
 * -------------------------------------------
 * https://numpy.org/doc/1.17/reference/c-api.array.html#c.PyArray_DIM
 */
void    *nd_data(t_ndarray *a)
{
	if (a == NULL)
		return NULL;

	return a->raw_data;
}

/*
 * Function: nd_step
 * --------------------
 * Return the stride in the nth dimension
 *
 * 	Parameters	:
 *		a 	  : python array object
 * 		index	  : dimension index
 * 	Returns		:
 *		return NULL if object is NULL or the data of the array
 * reference of the used c/numpy api function
 * -------------------------------------------
 * https://numpy.org/doc/1.17/reference/c-api.array.html#c.PyArray_DIM
 */
int     nd_nstep(t_ndarray *a, int n)
{
	if (a == NULL)
		return 0;

	int step = a->strides[n];
	for (int i = n+1; i<a->nd; ++i) {
		step /= a->shape[i];
	}
	return step;
}

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


/* converting numpy array to c nd array*/
t_ndarray	pyarray_to_ndarray(PyArrayObject *a)
{
	t_ndarray		array;

	array.nd          = PyArray_NDIM(a);
	array.raw_data    = PyArray_DATA(a);
	array.type_size   = PyArray_ITEMSIZE(a);
	array.type        = PyArray_TYPE(a);
	array.length      = PyArray_SIZE(a);
	array.buffer_size = PyArray_NBYTES(a);
	array.shape       = _numpy_to_ndarray_shape(PyArray_SHAPE(a), array.nd);
	array.strides     = _numpy_to_ndarray_strides(PyArray_STRIDES(a), array.type_size, array.nd);

	array.is_view     = 1;

	return array;
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
 *      index : dimension index
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

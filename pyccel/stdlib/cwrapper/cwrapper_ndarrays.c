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
 * Function: pyarray_to_c_ndarray
 * ----------------------------
 * A Cast function that convert numpy array variable into ndarray variable,
 * by copying its information and data to a new variable of type ndarray struct
 * and return this variable to be used inside c code.
 * Parameters :
 *     o : python array object
 *
 * Returns    :
 *     array : c ndarray
 *
 * reference of the used c/numpy api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html
 */
t_ndarray	pyarray_to_c_ndarray(PyArrayObject *a)
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

/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

/*
 * File containing functions useful for the cwrapper which require ndarrays.
 * There are 2 types of functions:
 * - Functions converting PythonObjects to standard C types
 * - Functions converting standard C types to PythonObjects (TODO: issue #537)
 */

#ifndef CWRAPPER_NDARRAYS_H
# define CWRAPPER_NDARRAYS_H

# include "cwrapper.h"
# include "ndarrays.h"

# define NO_TYPE_CHECK -1
# define NO_ORDER_CHECK -1

/*
 * Function: pyarray_to_ndarray
 * ----------------------------
 * A Cast function that converts a numpy array variable into a ndarray variable,
 * by copying its information and data pointer to a new variable of type
 * ndarray struct and return this variable to be used inside c code.
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
enum NPY_TYPES get_numpy_type(t_ndarray *o);
enum e_types get_ndarray_type(PyArrayObject *a);
/*
 * Function: pyarray_to_ndarray
 * ----------------------------
 * A Cast function that converts a ndarray variable into a numpy array variable,
 * by copying its information and data pointer to a new variable of type
 * PyObject and return this variable to be used inside python code.
 * Parameters :
 *     o : python array object
 *     release_data : bool indicating whether data is released for numpy to take
 *                      care of its memory cleanup
 *
 * Returns    :
 *     array : c ndarray
 *
 * reference of the used c/numpy api function
 * -------------------------------------------
 * https://numpy.org/doc/stable/reference/c-api/array.html
 */
t_ndarray	pyarray_to_ndarray(PyObject *o);
PyObject* ndarray_to_pyarray(t_ndarray *o, bool release_data);
PyObject* c_ndarray_to_pyarray(t_ndarray *o, bool release_data);
PyObject* fortran_ndarray_to_pyarray(t_ndarray *o, bool release_data);

/* arrays checkers and helpers */
bool	pyarray_check(PyObject *o, int dtype, int rank, int flag);
bool	is_numpy_array(PyObject *o, int dtype, int rank, int flag);

void    *nd_data(t_ndarray *a);
int     nd_ndim(t_ndarray *a, int n);

#endif

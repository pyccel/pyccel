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
t_ndarray	pyarray_to_ndarray(PyArrayObject *o);


/* arrays checkers and helpers */
bool	pyarray_check(PyArrayObject *o, int dtype, int rank, int flag);

void    *nd_data(t_ndarray *a);
int     nd_ndim(t_ndarray *a, int n);

#endif

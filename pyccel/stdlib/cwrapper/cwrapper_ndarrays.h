/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#ifndef CWRAPPER_NDARRAYS_H
# define CWRAPPER_NDARRAYS_H

# include "cwrapper.h"
# include "ndarrays.h"
/* converting numpy array to c nd array*/
t_ndarray	pyarray_to_c_ndarray(PyArrayObject *o);

#endif

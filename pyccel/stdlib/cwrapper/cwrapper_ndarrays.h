#ifndef CWRAPPER_NDARRAYS_H
# define CWRAPPER_NDARRAYS_H

# include "cwrapper.h"
# include "ndarrays.h"
/* converting numpy array to c nd array*/
t_ndarray	pyarray_to_c_ndarray(PyArrayObject *o);

#endif

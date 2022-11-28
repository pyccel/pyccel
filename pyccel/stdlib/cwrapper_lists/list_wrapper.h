
#ifndef LST_WRAPPER
#define LST_WRAPPER
// # include "../cwrapper/cwrapper.h"

#include "../lists/lists.h"
# include "Python.h"
# include <complex.h>
# include <stdint.h>
# include <stdbool.h>

t_list* unwrap_list(PyObject *list);
PyObject * wrap_list(t_list *list);
bool	pylist_check(PyListObject *o, int dtype);


#endif

#ifndef LST_WRAPPER
#define LST_WRAPPER
# include "cwrapper.h"
#include "lists.h"

t_list* unwrap_list(PyObject *self, PyObject *list);
PyObject * wrap_list(t_list *list);


#endif
#ifndef LISTS_WRAPPER_TOOLS_H
#define LISTS_WRAPPER_TOOLS_H

#include "lists.h"

int PyObject_to_PyccelList(PyObject *pylist, PyccelList **pyccel_list);
int PyccelList_to_PyObject(PyccelList *pyccel_list, PyObject **pylist);

#endif

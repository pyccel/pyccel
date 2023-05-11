#include "lists_wrapper_tools.h"


int     PyObject_to_PyccelList(PyObject *py_list, PyccelList **pyccel_list)
{
    Py_ssize_t py_list_size;
    PyObject *pyob_item;
    size_t pyitem;

    py_list_size = PyList_Size(py_list);
    for (int i = 0; i < py_list_size;i++)
    {
        pyob_item = PyList_GetItem(py_list, i);
        if (PyList_Check(pyob_item))
        {
            (*pyccel_list)->items[i]->pointer = initialise_list(PyList_Size(pyob_item));
            (*pyccel_list)->items[i]->type = "pl";
            PyObject_to_PyccelList(pyob_item, (PyccelList **)(&((*pyccel_list)->items[i]->pointer)));
        }
        else
        {
            pyitem = (size_t)PyLong_AsLongLong(pyob_item);
            initialise_genericobject(&((*pyccel_list)->items[i]), "v", pyitem, NULL);
        }
    }
    return 0;
}

int PyccelList_to_PyObject(PyccelList *pyccel_list, PyObject **pylist)
{
    size_t pyccel_list_size;
    size_t pyccel_list_item;
    PyccelList *pyccel_sublist;
    PyObject *pylist_tmp;
    
    pyccel_list_size = pyccel_list->noi;
    for (size_t i = 0; i < pyccel_list_size; i++)
    {
        if (!strcmp(pyccel_list->items[i]->type, "pl"))
        {
            pyccel_sublist = (PyccelList *)pyccel_list->items[i]->pointer;
            pylist_tmp = PyList_New(pyccel_sublist->noi);
            PyList_SetItem(*pylist, i, pylist_tmp);
            PyccelList_to_PyObject(pyccel_sublist, &pylist_tmp);
        }
        else
        {
            pyccel_list_item = pyccel_list->items[i]->value;
            PyList_SetItem(*pylist, i, PyLong_FromSize_t(pyccel_list_item));
        }
    }

    return 0;
}
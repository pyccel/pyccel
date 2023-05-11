#include "lists.h"
#include "lists_wrapper_tools.h"


static PyObject *pyccel_append_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    size_t item;
    PyccelList *pyccel_list_arg = NULL;
    PyccelList *pyccel_list = NULL;
    PyObject *item_tmp;
    PyObject *pylist_tmp;
    PyObject *newpylist;
    static char *kwlist[] = {
        "list",
        "item",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &pylist_tmp, &item_tmp))
    {
        return NULL;
    }
    if (!PyList_Check(pylist_tmp))
    {
        PyErr_SetString(PyExc_TypeError, "\"First argument must be a list\"");
        return NULL;
    }
    if (PyLong_CheckExact(item_tmp))
    {
        item = (size_t)PyLong_AsLongLong(item_tmp);
    }
    else if (PyList_Check(item_tmp))
    {
        size_t item_tmp_size = PyList_Size(item_tmp);
        pyccel_list_arg = initialise_list(item_tmp_size);
        PyObject_to_PyccelList(item_tmp, &pyccel_list_arg);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Second argument must be native int or a list\"");
        return NULL;

    }

    size_t pylist_size = PyList_Size(pylist_tmp);
    pyccel_list = initialise_list(pylist_size);
    PyObject_to_PyccelList(pylist_tmp, &pyccel_list);

    if (pyccel_list_arg == NULL)
        pyccel_append(&pyccel_list, item, NULL);
    else
        pyccel_append(&pyccel_list, 0, pyccel_list_arg);

    size_t pyccel_list_size = pyccel_list->noi;
    newpylist = PyList_New(pyccel_list_size);
    PyccelList_to_PyObject(pyccel_list, &newpylist);

    free_list(pyccel_list);
    return newpylist;
}


static PyMethodDef pyccel_append_methods[] = {
    {
        "pyccel_append",
        (PyCFunction)pyccel_append_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        ""
    },
    { NULL, NULL, 0, NULL}
};


static struct PyModuleDef pyccel_append_module = {
    PyModuleDef_HEAD_INIT,
    /* name of module */
    "pyccel_append",
    /* module documentation, may be NULL */
    NULL,
    /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    -1,
    pyccel_append_methods,
};

PyMODINIT_FUNC PyInit_pyccel_append(void)
{
    return PyModule_Create(&pyccel_append_module);
}

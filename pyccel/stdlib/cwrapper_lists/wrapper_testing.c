#include "list_wrapper.h"

static PyObject* py_extend(PyObject* self, PyObject* args)
{
    PyObject *list1 = NULL;
    PyObject *list2 = NULL;
 
    if (!PyArg_ParseTuple(args, "o", &list1))
        return NULL;
    if (!PyArg_ParseTuple(args, "o", &list2))
        return NULL;
    t_list *clist1 = unwrap_list(list1);
    t_list *clist2 = unwrap_list(list2);
    extend(clist1, clist2);
    return wrap_list(clist1);
}

static PyObject* py_clear(PyObject* self, PyObject* args)
{
    PyObject *list = NULL;
 
    if (!PyArg_ParseTuple(args, "o", &list))
        return NULL;

    t_list *clist = unwrap_list(list);
    clear(clist);
    return NULL;
}

static PyObject* py_copy(PyObject* self, PyObject* args)
{
    PyObject *list = NULL;
 
    if (!PyArg_ParseTuple(args, "o", &list))
        return NULL;

    t_list *clist = unwrap_list(list);
    t_list *clist_copy = copy(clist);
    return wrap_list(clist_copy);
}

static PyMethodDef myMethods[] = {
    {"extend", py_extend, METH_VARARGS, "extend the first list with the."},
    {"clear", py_clear, METH_VARARGS, "clear list."},
    {"clear", py_copy, METH_VARARGS, "copy list from another one."},
    {NULL, NULL, 0, NULL}
};
 
static struct PyModuleDef listModule = {
	PyModuleDef_HEAD_INIT,
	"lists",
	"basic list suppport for pyccel",
	-1,
	myMethods
};

PyMODINIT_FUNC PyInit_listModule(void)
{
    return PyModule_Create(&listModule);
}

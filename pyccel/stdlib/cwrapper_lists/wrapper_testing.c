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
 
static PyMethodDef myMethods[] = {
    {"extend", py_extend, METH_VARARGS, "extend the first list with the ."},
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

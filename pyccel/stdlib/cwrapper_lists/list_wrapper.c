#include "list_wrapper.h"


char *typeStr[5] = {"p", "i", "f", "d"};//TODO: double check these.


static  void* pylist_get_literal(PyObject * item, size_t type)  // check if there is a better way than allocating a buffer
{
    Py_ssize_t tsize = tSizes[type];
    char * cItem = malloc(tsize);

    memset(cItem, 0, tsize);
    if (type == lst_bool)
        *cItem = (PyObject_IsTrue(item)) ? 1 : 0;
    else if (type == lst_int64)
    {
        int64_t int_v = PyLong_AsLong(item);
        memcpy(cItem, &int_v, tsize);
    }
    else if (type == lst_float || type == lst_double)
    {
        double double_v = PyFloat_AsDouble(item);
        memcpy(cItem, &double_v, tsize);
    }
    return cItem;
}

static void *pylist_get_elements(PyObject* list, t_type type, size_t size)
{
    Py_ssize_t  tsize = tSizes[type];
    char *elements = malloc(size * tsize);
    Py_ssize_t index = 0;

    while (index < size)
    {
        if (type == lst_list)
            memcpy(&(elements[index * tsize]), unwrap_list(((PyListObject*)list)->ob_item[index]), tsize);
        else
        {
            char * item = pylist_get_literal(((PyListObject*)list)->ob_item[index], type);
            memcpy(&(elements[index * tsize]), item, tsize);
            free(item);
        }
        index++;
    }
    return elements;
}

static t_type pylist_get_type(PyObject *list, Py_ssize_t index)
{
    PyObject *element = PyList_GetItem(list, index);
    if (PyList_CheckExact(element))
        return lst_list;
    else if (PyLong_CheckExact(element))
        return lst_int64;
    else if (PyFloat_CheckExact(element))
        return lst_float;
    else if (PyBool_Check(element))
        return lst_bool;
    return lst_none;
}

t_list* unwrap_list(PyObject *list)
{
    if (!PyList_CheckExact(list))
        return (NULL);
    Py_ssize_t size = PyList_Size(list);
    t_type type = lst_none;
    if (size)
        type = pylist_get_type(list, 0);
    char *elements = pylist_get_elements(list, type, size);
    return allocate_list(size, type, elements);
}

PyObject * wrap_list(t_list *list)
{
    Py_ssize_t size = list->size;
    Py_ssize_t index = 0;
    PyObject * pylist = PyList_New(size);
    PyObject * obj = NULL;

    while (index < size)
    {
        if (list->type == lst_list)
            obj = Py_BuildValue(typeStr[list->type], wrap_list(list));
        else
            obj = Py_BuildValue(typeStr[list->type], array_subscripting(list, index));
        PyList_SetItem(pylist, index, obj);
        index++;
    }
    return (pylist);
}

bool    pylist_check_elements_type(PyListObject *list, t_type dtype)
{
    Py_ssize_t index = 0;
    while (index < list->size)
    {
        if (pylist_get_type(list, dtype) != dtype)
        return false;
        index++;
    }
    return true;
}

bool	pylist_check(PyListObject *list, t_type dtype)
{
	if (!PyList_CheckExact(list))
        return false;
    Py_ssize_t list_size = PyList_Size(list);
    Py_ssize_t  index = 0;
    if (list_size == 0)
        return true;
    return pylist_check_elements_type(list);
}
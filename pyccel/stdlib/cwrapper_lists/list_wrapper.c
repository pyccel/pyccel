#include "list_wrapper.h"


char *typeStr[5] = {"p", "i", "f", "d"};//TODO: double check these.


static  void* pylist_get_literal(PyObject * item, size_t type)  // check if there is a better way than allocating a buffer
{
    size_t tsize = tSizes[type];
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
    size_t  tsize = tSizes[type];
    char *elements = malloc(size * tsize);
    size_t index = 0;

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

static t_type pylist_get_type(PyObject *list)
{
    PyObject *element = PyList_GetItem(list, 0);
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
    size_t size = PyList_Size(list);
    t_type type = lst_none;
    if (size)
        type = pylist_get_type(list);
    char *elements = pylist_get_elements(list, type, size);
    return allocate_list(size, type, elements);
}

PyObject * wrap_list(t_list *list)
{
    size_t size = list->size;
    size_t index = 0;
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

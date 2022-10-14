#include "list_wrapper.h"


char *typeStr[5] = {"b", "i", "f", "d", "p"};//TODO: double check these.

static void *pylist_get_elements(PyObject* list, t_type type, size_t size)
{
    size_t  tsize = tSizes[type];
    char *elements = malloc(size * tsize);
    size_t index = 0;

    while (index < size)
    {
        if (type == lst_list)
            memcpy(&(elements[index * tsize]), list_wrapper(list->ob_item[index]), tsize);
        else
        {
            char * item = pylist_get_literal(list->ob_item[index], type);
            memcpy(&(elements[index * tsize]), item, tsize);
            free(item);
        }
        index++;
    }
    return elements;
}

static t_list* unwrap_list(PyObject *self, PyObject *list)
{
    if (!PyList_CheckExact(list))
        return (NULL);
    size_t size = list.ob_size;
    t_type type = lst_none;
    if (size)
        type = pylist_get_type(list);
    char *elements = pylist_get_elements(list, type, size);
    return allocate_list(size, type, elements);
}


static t_type pylist_get_type(Py_LISTObject *list)
{
    PyObject *element = PyList_GetItem(list);
    if (PyList_CheckExact(element))
        return lst_list;
    else if (PyLong_CheckExact(element))
        return lst_int;
    else if (PyFloat_CheckExact(element))
        return lst_float;
    else if (PyBool_Check(element))
        return lst_bool;
    return lst_none;
}

static  void* pylist_get_literal(PyObject * item, size_t type) // check if there is a better way than allocating a buffer
{
    size_t tsize = tSizes[type];
    char * cItem = malloc(tsize);
    memset(cItem, 0, tsize);
    if (type == lst_bool)
        *cItem = (PyObject_IsTrue(item)) ? 1 : 0;
    else if (type == lst_int)
        memcpy(cItem, &PyLong_AsLong(item), tsize);
    else if (type == lst_float)
        memcpy(cItem, &PyFloat_AsDouble(item), tsize);
    else if (type == lst_double)
        memcpy(cItem, &PyFloat_AsDouble(item), tsize);//redundant
    return cItem;
}

PyObject * wrap_list(t_list list)
{
    size_t size = list->size;
    size_t index = 0;
    Py_ListObject * pylist = PyList_New(size);
    pyObject * obj = NULL;

    while (index < size)
    {
        obj = Py_BuildValue(typeStr[list->type], array_subscripting(list, index));
        PyList_SetItem(list, index, obj);
        index++;
    }
    return (pylist);
}
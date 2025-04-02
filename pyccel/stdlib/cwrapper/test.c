#include "cwrapper.h"

static PyObject* bind_c_f12_wrapper(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* x_obj;
    PyObject* y_obj;
    void* x_data;
    int64_t x_shape[1];
    int64_t x_strides[1];
    void* y_data;
    int64_t y_shape[1];
    int64_t y_strides[1];
    y_obj = Py_None;
    static char *kwlist[] = {
        "x",
        "y",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &x_obj, &y_obj))
    {
        return NULL;
    }
    if (pyarray_check(x_obj, NPY_LONG, INT64_C(1), NO_ORDER_CHECK))
    {
        x_data = PyArray_DATA((PyArrayObject*)(x_obj));
        get_strides_and_shape_from_numpy_array(x_obj, x_shape, x_strides);
    }
    else
    {
        return NULL;
    }
    printf("%s %s\n", "Is PyNone : ", y_obj == Py_None ? "True" : "False");
    if (y_obj == Py_None)
    {
        y_data = NULL;
    }
    else
    {
        printf("%s\n", "Detected not PyNone");
        if (pyarray_check(y_obj, NPY_LONG, INT64_C(1), NO_ORDER_CHECK))
        {
            y_data = PyArray_DATA((PyArrayObject*)(y_obj));
            get_strides_and_shape_from_numpy_array(y_obj, y_shape, y_strides);
        }
        else
        {
            return NULL;
        }
    }
    bind_c_f12(x_data, x_shape[INT64_C(0)], x_strides[INT64_C(0)], y_data, y_shape[INT64_C(0)], y_strides[INT64_C(0)]);
    Py_INCREF(Py_None);
    return Py_None;
}

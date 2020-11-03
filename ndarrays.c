#include "ndarrays.h"

/* 
** allocation
*/

t_ndarray   array_create(int nd, int *shape, enum e_types type)
{
    t_ndarray arr;

    arr.nd = nd;
    arr.type = type;
    switch (type)
    {
        case nd_int:
            arr.type_size = sizeof(int);
            break;
        case nd_float:
            arr.type_size = sizeof(float);
            break;
        case nd_double:
            arr.type_size = sizeof(double);
            break;
        case nd_cdouble:
            arr.type_size = sizeof(double complex);
            break;
    }
    arr.is_slice = false;
    arr.length = 1;
    arr.shape = malloc(arr.nd * sizeof(int));
    for (int i = 0; i < arr.nd; i++)
    {
        arr.length *= shape[i];
        arr.shape[i] = shape[i];
    }
    arr.buffer_size = arr.length * arr.type_size;
    arr.strides = malloc(nd * sizeof(int));
    for (int i = 0; i < arr.nd; i++)
    {
        arr.strides[i] = 1;
        for (int j = i + 1; j < arr.nd; j++)
            arr.strides[i] *= arr.shape[j];
    }
    arr.raw_data = malloc(arr.buffer_size);
    return (arr);
}

/*
** deallocation
*/

int free_array(t_ndarray dump)
{
    if (!dump.is_slice)
    {
        free(dump.raw_data);
        dump.raw_data = NULL;
    }
    free(dump.shape);
    dump.shape = NULL;
    free(dump.strides);
    dump.strides = NULL;
    return (1);
}

/* 
** slices
*/

t_slice new_slice(int start, int end, int step)
{
    t_slice slice_d;

    slice_d.start = start;
    slice_d.end = end;
    slice_d.step = step;
    return (slice_d);
}

t_ndarray array_slicing(t_ndarray p, ...)
{
    t_ndarray slice;
    va_list  va;
    t_slice slice_data;
    int start = 0;

    slice.nd = p.nd;
    slice.type = p.type;
    slice.type_size = p.type_size;
    slice.shape = malloc(sizeof(int) * p.nd);
    slice.strides = malloc(sizeof(int) * p.nd);
    memcpy(slice.strides, p.strides, sizeof(int) * p.nd);
    slice.is_slice = true;
    va_start(va, p);
    for (int i = 0; i < p.nd ; i++)
    {
        slice_data = va_arg(va, t_slice);
        slice.shape[i] = (slice_data.end - slice_data.start + (slice_data.step - 1)) / slice_data.step; // we need to round up the shape
        start += slice_data.start * p.strides[i];
        slice.strides[i] *= slice_data.step;
    }
    va_end(va);
    slice.raw_data = p.raw_data + start * p.type_size;
    slice.length = 1;
    for (int i = 0; i < slice.nd; i++)
            slice.length *= slice.shape[i];
    return (slice);
}

/*
** indexing
*/

int get_index(t_ndarray arr, ...)
{
    va_list va;
    int index;

    va_start(va, arr);
    index = 0;
    for (int i = 0; i < arr.nd; i++)
    {
        index += va_arg(va, int) * arr.strides[i];
    }
    va_end(va);
    return (index);
}

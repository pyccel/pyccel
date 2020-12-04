#include "ndarrays.h"

/* 
** allocation
*/

t_ndarray   array_create(int32_t nd, int32_t *shape, enum e_types type)
{
    t_ndarray arr;

    arr.nd = nd;
    arr.type = type;
    switch (type)
    {
        case nd_int8:
            arr.type_size = sizeof(int8_t);
            break;
        case nd_int16:
            arr.type_size = sizeof(int16_t);
            break;
        case nd_int32:
            arr.type_size = sizeof(int32_t);
            break;
        case nd_int64:
            arr.type_size = sizeof(int64_t);
            break;
        case nd_float:
            arr.type_size = sizeof(float);
            break;
        case nd_double:
            arr.type_size = sizeof(double);
            break;
        case nd_bool:
            arr.type_size = sizeof(bool);
            break;
        case nd_cfloat:
            arr.type_size = sizeof(float complex);
            break;
        case nd_cdouble:
            arr.type_size = sizeof(double complex);
            break;
    }
    arr.is_slice = false;
    arr.length = 1;
    arr.shape = malloc(arr.nd * sizeof(int32_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.length *= shape[i];
        arr.shape[i] = shape[i];
    }
    arr.buffer_size = arr.length * arr.type_size;
    arr.strides = malloc(nd * sizeof(int32_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.strides[i] = 1;
        for (int32_t j = i + 1; j < arr.nd; j++)
            arr.strides[i] *= arr.shape[j];
    }
    arr.raw_data = malloc(arr.buffer_size);
    return (arr);
}

void   _array_fill_int8(int8_t c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_int8[i] = c;
}

void   _array_fill_int16(int16_t c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_int16[i] = c;
}

void   _array_fill_int32(int32_t c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_int32[i] = c;
}

void   _array_fill_int64(int64_t c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_int64[i] = c;
}

void   _array_fill_bool(bool c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_bool[i] = c;
}

void   _array_fill_float(float c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_float[i] = c;
}

void   _array_fill_double(double c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_double[i] = c;
}

void   _array_fill_cfloat(float complex c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_cfloat[i] = c;
}


void   _array_fill_cdouble(double complex c, t_ndarray arr)
{
    if (c == 0)
        memset(arr.raw_data, 0, arr.buffer_size);
    else
        for (int32_t i = 0; i < arr.length; i++)
            arr.nd_cdouble[i] = c;
}

/*
** deallocation
*/

int32_t free_array(t_ndarray dump)
{
    if (dump.shape == NULL)
        return (0)
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

t_slice new_slice(int32_t start, int32_t end, int32_t step)
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
    int32_t start = 0;

    slice.nd = p.nd;
    slice.type = p.type;
    slice.type_size = p.type_size;
    slice.shape = malloc(sizeof(int32_t) * p.nd);
    slice.strides = malloc(sizeof(int32_t) * p.nd);
    memcpy(slice.strides, p.strides, sizeof(int32_t) * p.nd);
    slice.is_slice = true;
    va_start(va, p);
    for (int32_t i = 0; i < p.nd ; i++)
    {
        slice_data = va_arg(va, t_slice);
        slice.shape[i] = (slice_data.end - slice_data.start + (slice_data.step - 1)) / slice_data.step; // we need to round up the shape
        start += slice_data.start * p.strides[i];
        slice.strides[i] *= slice_data.step;
    }
    va_end(va);
    slice.raw_data = p.raw_data + start * p.type_size;
    slice.length = 1;
    for (int32_t i = 0; i < slice.nd; i++)
            slice.length *= slice.shape[i];
    return (slice);
}

/*
** indexing
*/

int32_t get_index(t_ndarray arr, ...)
{
    va_list va;
    int32_t index;

    va_start(va, arr);
    index = 0;
    for (int32_t i = 0; i < arr.nd; i++)
    {
        index += va_arg(va, int32_t) * arr.strides[i];
    }
    va_end(va);
    return (index);
}

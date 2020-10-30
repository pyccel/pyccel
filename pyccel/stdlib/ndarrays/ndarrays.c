#include "ndarrays.h"

/* 
** allocation
*/

t_ndarray array_init(char *temp, int nd, int *shape, enum e_types type, int type_size)
{
    t_ndarray a;

    a.type = type;
    a.type_size = type_size;
    a.nd = nd;
    a.shape = malloc(a.nd * sizeof(int));
    a.length = 1;
    a.is_slice = false;
    for (int i = 0; i < a.nd; i++) // init the shapes
    {
        a.length *= shape[i];
        a.shape[i] = shape[i];
    }
    a.strides = malloc(nd * sizeof(int));
    for (int i = 0; i < a.nd; i++)
    {
        a.strides[i] = 1;
        for (int j = i + 1; j < a.nd; j++)
            a.strides[i] *= a.shape[j];
    }
    a.raw_data = calloc(a.length , a.type_size);
    if (temp)
        memcpy(a.raw_data, temp, a.length * a.type_size);
    return (a);
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

t_slice slice_data(int start, int end, int step)
{
    t_slice slice_d;

    slice_d.start = start;
    slice_d.end = end;
    slice_d.step = step;
    return (slice_d);
}

t_ndarray slice_make(t_ndarray p, ...)
{
    t_ndarray slice;
    va_list  va;
    t_slice slice_data;
    t_slice s;
    int i = 0;
    int start = 0;

    slice.nd = p.nd;
    slice.type = p.type;
    slice.type_size = p.type_size;
    slice.shape = malloc(sizeof(int) * p.nd);
    memcpy(slice.shape, p.shape, sizeof(int) * p.nd);
    slice.strides = malloc(sizeof(int) * p.nd);
    memcpy(slice.strides, p.strides, sizeof(int) * p.nd);
    slice.is_slice = true;
    va_start(va, p);
    while (i < p.nd)
    {
        slice_data = va_arg(va, t_slice);
        slice.shape[i] = (slice_data.end - slice_data.start + (slice_data.step / 2)) / slice_data.step; // we need to round up the shape
        start += slice_data.start * p.strides[i];
        slice.strides[i] *= slice_data.step;
        i++;
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
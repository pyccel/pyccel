#include "ndarray.h"

t_slice *slice_data(int start, int end, int step)
{
    t_slice *slice_d;

    slice_d = malloc(sizeof(t_slice));
    slice_d->start = start;
    slice_d->end = end;
    slice_d->step = step;

    return (slice_d);
}

t_ndarray *make_slice(t_ndarray *p, ...)
{
    t_ndarray *slice;
    va_list     va;
    t_slice     *slice_data;
    int i = 0;
    int start = 0;

    slice = malloc(sizeof(t_ndarray));
    slice->nd = p->nd;
    slice->type = p->type;
    slice->shape = malloc(sizeof(int) * p->nd);
    memcpy(slice->shape, p->shape, sizeof(int) * p->nd);
    slice->strides = malloc(sizeof(int) * p->nd);
    memcpy(slice->strides, p->strides, sizeof(int) * p->nd);
    slice->is_slice = 1;
    va_start(va, p);
    while ((slice_data = va_arg(va, t_slice*)))
    {
        slice->shape[i] = (slice_data->end - slice_data->start + (slice_data->step / 2)) / slice_data->step;
        start += slice_data->start * p->strides[i];
        slice->strides[i] *= slice_data->step;
        i++;
        free(slice_data);
    }
    va_end(va);
    slice->data = malloc(sizeof(t_ndarray_type));
    slice->data->raw_data = p->data->raw_data + start * p->type;
    slice->lenght = 1;
    for (int i = 0; i < slice->nd; i++)
            slice->lenght *= slice->shape[i];
    return (slice);
}

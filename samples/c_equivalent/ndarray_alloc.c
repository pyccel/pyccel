#include "ndarray.h"

t_ndarray *init_array(char *temp, int nd, int *shape, int type)
{
    t_ndarray *a;
    
    a = malloc(sizeof(t_ndarray));
    a->type = type;
    a->nd = nd;
    a->shape = malloc(a->nd * sizeof(int));
    a->lenght = 1;
    a->is_slice = 0;
    for (int i = 0; i < a->nd; i++) // init the shapes
    {
        a->lenght *= shape[i]; 
        a->shape[i] = shape[i];
    }
    a->strides = malloc(nd * sizeof(int));
    for (int i = 0; i < a->nd; i++)
    {
        a->strides[i] = 1;
        for (int j = i + 1; j < a->nd; j++)
            a->strides[i] *= a->shape[j];
    }
    a->data = malloc(sizeof(t_ndarray_type));
    a->data->raw_data = calloc(a->lenght , a->type);
    if (temp)
        memcpy(a->data->raw_data, temp, a->lenght * a->type);
    return (a);
}
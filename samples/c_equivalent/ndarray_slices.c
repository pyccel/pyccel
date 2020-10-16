#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ndarray.h"
#include <time.h>
#include <stdarg.h>

int get_index(t_ndarray *arr, ...)
{
    va_list va;
    int index;
    va_start(va, arr);

    index = 0;
    for (int i = 0; i < arr->nd; i++)
    {
            index += va_arg(va, int) * arr->strides[i];
    }
    va_end(va);
    return (index);
}

t_slice *slice_data(int start, int end, int step)
{
    t_slice *slice_d;

    slice_d = malloc(sizeof(t_slice));
    slice_d->start = start;
    slice_d->end = end;
    slice_d->step = step;

    return (slice_d);
}

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

int free_array(t_ndarray *dump)
{
    if (!dump->is_slice)
    {
        free(dump->data->raw_data);
        dump->data->raw_data = NULL;
        free(dump->data);
        dump->data = NULL;
    }
    free(dump->shape);
    dump->shape = NULL;
    free(dump->strides);
    dump->strides = NULL;
    free(dump);
    dump = NULL;
    return (1);
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

int array_value_dump(t_ndarray *arr)
{
    switch (arr->nd)
    {
        case 1:
            for (int i = 0; i < arr->lenght; i++)
            {
                printf(" %f,", arr->data->double_nd[get_index(arr, i)]);
            }
            putchar('\n');
            break;
        case 2:
            for (int i = 0; i < arr->shape[0]; i++)
            {
                for (int j = 0; j < arr->shape[1]; j++)
                    printf(" %f,", arr->data->double_nd[get_index(arr, i, j)]);
                putchar('\n');  
            }
            putchar('\n');  
            break;
        case 3:
            for (int i = 0; i < arr->shape[0]; i++)
            {
                for (int j = 0; j < arr->shape[1]; j++)
                {
                    for (int k = 0; k < arr->shape[2]; k++)
                        printf(" %f,", arr->data->double_nd[get_index(arr, i, j, k)]);
                    putchar('\n');  
                }
                putchar('\n');  
                putchar('\n');  
            }
            putchar('\n');  
            break;
        default:
            break;
    }
    return (1);
}
int array_data_dump(t_ndarray *arr)
{
    int a;
    printf("array : \n\tndim %d\n\ttype %d\n\tlenght %d\n", arr->nd, arr->type, arr->lenght);
    printf(" %d - %d - %d\n", arr->shape[0], arr->shape[1], arr->shape[2]);
    array_value_dump(arr);
    return (1);
}


int main(void)
{
    int i;
    double m_1[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 8, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1, 200, 33, 5, 57, 62, 70, 103, 141, 122, 26.50, 36.334, 82, 8.44002, 10.056, 4115, 22.1, 1.1102, 011.25, 1.01110005, 19};
    int m_1_shape[] = {2, 4, 5};

    t_ndarray *x;
    t_ndarray *y;

    /* init the fist matrix */
    x = init_array((char *)m_1, 3, m_1_shape, sizeof(double));
    array_data_dump(x);
    y = NULL;
    y = make_slice(x, slice_data(0, 2, 1), slice_data(1, 3, 1), slice_data(1, 3, 1), NULL);
    printf("\n\n");
    array_data_dump(y);
    y->data->double_nd[get_index(y, 1, 0, 0)] = 100.2;
    array_data_dump(y);
    array_data_dump(x);
    free_array(x);
    free_array(y);
    return (0);
}

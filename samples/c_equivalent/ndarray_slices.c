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
        if (arr->slices && arr->slices[i])
            index += ((va_arg(va, int) + arr->slices[i]->start) * arr->strides[i]);
        else
        {
            index += va_arg(va, int) * arr->strides[i];
        }
    }
    va_end(va);
    return (index);
}

t_ndarray *init_array(char *temp, int nd, int *shape, int type)
{
    t_ndarray *a;
    
    a = malloc(sizeof(t_ndarray));
    a->type = type;
    a->nd = nd;
    a->shape = malloc(a->nd * sizeof(int));
    a->lenght = 1;
    for (int i = 0; i < a->nd; i++) // init the shapes
    {
        a->lenght *= shape[i]; 
        a->shape[i] = shape[i];
    }
    a->slices = NULL; //
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
    if (!dump->slices)
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
    if (dump->slices)
    {
        for (int i = 0; i < dump->nd; i++)
        {
            free(dump->slices[i]);
        }
        free(dump->slices);
        dump->slices = NULL;
    }
    free(dump);
    dump = NULL;
    return (1);
}

t_ndarray *make_slice(t_ndarray *slice, t_ndarray *p, int dim, int start, int end, int step)
{
    if (!slice)
    {
        slice = malloc(sizeof(t_ndarray));
        slice->data = p->data;
        slice->nd = p->nd;
        slice->type = p->type;
        slice->strides = malloc(sizeof(int) * p->nd);
        memcpy(slice->strides, p->strides, sizeof(int) * p->nd);
        slice->shape = malloc(sizeof(int) * p->nd);
        memcpy(slice->shape, p->shape, sizeof(int) * p->nd);
        slice->slices = NULL;
    }
    // printf("dim %d - p shape %d - %d\n", dim , p->shape[dim] , p->shape[dim] - (end - start));
    /* slice data */
    if (!slice->slices)
        slice->slices = calloc(sizeof(t_slice *) , p->nd);
    slice->slices[dim] = malloc(sizeof(t_slice));
    slice->slices[dim]->end = end;
    slice->slices[dim]->step = step;
    slice->slices[dim]->start = start;
    slice->shape[dim] = (end - start);
    slice->strides[dim] = p->strides[dim] * step;
    slice->lenght = 1;
    for (int i = 0; i < slice->nd; i++)
    {
        if (slice->slices && slice->slices[i])
        {
            slice->lenght *= slice->shape[i];
        }
        else
        {
            slice->lenght *= p->shape[i];
        }
    }
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
    array_value_dump(arr);
    return (1);
}


int main(void)
{
    int i;
    double m_1[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 8, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1};
    int m_1_shape[] = {4, 5};

    t_ndarray *x;
    t_ndarray *y;

    /* init the fist matrix */
    x = init_array((char *)m_1, 2, m_1_shape, sizeof(double));
    /* the product matrix */
    // printf("x->shape : (%d) - x->strides : (%d)\n", x->shape[0], x->strides[0]);
    /* printing the result of the product */
    // i = 0;
    // while (i <  x->shape[0])
    // {
    //     printf(" %f,", x->data->double_nd[i]);
    //     i = i + 1;
    //     if (i % x->shape[0] == 0) // skipping a line when acessing the next row
    //         printf("\n");
    // }
    array_data_dump(x);
    y = NULL;
    y = make_slice(y, x, 0, 1, 3, 1);
    printf("\n\n");
    y = make_slice(y, x, 1, 0, 5, 1);
    array_data_dump(y);
    y->data->double_nd[get_index(y, 0, 0)] = 100.2;
    array_data_dump(y);
    array_data_dump(x);
    // printf("slice created\n");
    // printf("y shape %d\n", y->shape[0]);
    // i = 0;
    // while (i <  y->shape[0])
    // {
    //     printf(" %f,", y->data->double_nd[get_index(y, i)]);
    //     i = i + 1;
    //     if (i % y->shape[0] == 0) // skipping a line when acessing the next row
    //         printf("\n");
    // }
    free_array(x);
    free_array(y);
    return (0);
}

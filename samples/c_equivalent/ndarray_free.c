#include "ndarray.h"

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
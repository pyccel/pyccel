#include "cuda_ndarrays.h"

void *cuda_array_create(int shape[])
{
    size_t i = 0;
    size_t alloc_size = 1;

    while (shape[i] != 0)
    {
        alloc_size *= shape[i];
        i++;
    }

    void *array_ptr = malloc(alloc_size);
    if (array_ptr == NULL)
    {
        cout << "Error allocating memory" << endl;
        return NULL;
    }

    return array_ptr;
}
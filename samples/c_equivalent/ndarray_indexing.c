#include "ndarray.h"

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

int	get_index_o(t_ndarray *arr, int o)
{
	int index;
	int i;

	index = 0;
	for (i = 0; i < arr->nd; i++)
		if (o < arr->shape[i])
			break;
	index += o * arr->strides[i];
	return (index);
}
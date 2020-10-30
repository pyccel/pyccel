#include "ndarrays.h"

/* 
** allocation
*/

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

/* 
** deallocation
*/

int free_array(t_ndarray *dump)
{
	if (!dump->is_slice)
	{
		free(dump->data->raw_data);
		dump->data->raw_data = NULL;
	}
	free(dump->data);
	dump->data = NULL;
	free(dump->shape);
	dump->shape = NULL;
	free(dump->strides);
	dump->strides = NULL;
	free(dump);
	return (1);
}

/* 
** slices
*/

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
	va_list	 va;
	t_slice	 *slice_data;
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

/* 
** indexing
*/

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
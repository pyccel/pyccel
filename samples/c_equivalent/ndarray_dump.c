#include "ndarray.h"

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
	// for (int i = 0; i < arr->lenght; i++)
	// {
	// 	printf(" %f,", arr->data->double_nd[get_index_o(arr, i)]);
	// }
	array_value_dump(arr);
	return (1);
}

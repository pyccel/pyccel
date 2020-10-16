#include "../ndarray.h"

int main(void)
{
	int i;
	double m_1[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 8, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1, 200, 33, 5, 57, 62, 70, 103, 141, 122, 26.50, 36.334, 82, 8.44002, 10.056, 4115, 22.1, 1.1102, 011.25, 1.01110005, 19};
	int m_1_shape[] = {5, 8};

	t_ndarray *x;
	t_ndarray *y;

	/* init the fist matrix */
	x = init_array((char *)m_1, 1, m_1_shape, sizeof(double));
	array_data_dump(x);
	y = make_slice(x, slice_data(0, 2, 1), slice_data(0, 8, 1), NULL);
	printf("\n\n");
	// array_data_dump(y);
	// y->data->double_nd[get_index(y, 1, 0, 0)] = 100.2;
	// array_data_dump(y);
	// array_data_dump(x);
	free_array(x);
	// free_array(y);
	return (0);
}
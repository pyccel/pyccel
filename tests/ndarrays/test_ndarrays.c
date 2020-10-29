#include "ndarrays.h"
#include <unistd.h>


#define m_assert(X, Y, dscr) my_assert(X , Y, dscr, __func__, __FILE__, __LINE__)

#define my_assert(X , Y, dscr, __func__, __FILE__, __LINE__) _Generic((X), double: assert_double,\
							float: assert_float,\
							int: assert_int,\
							double complex : assert_complex_double,\
							default: assert_ns)(X , Y, dscr, __func__, __FILE__, __LINE__)
#define getname(X) #X

void assert_double(double v1 , double v2, const char *dscr, const char * func, const char *file, int line)
{

	if (v1 != v2)
	{
		printf("[FAIL] %s:%d:%s\n", file, line, func);
		printf("[INFO] %s:%f != %s:%f\n", getname(v1), v1, getname(v2),v2);
		printf("[DSCR] %s\n", dscr);
		return ;
	}
	printf("[PASS] %s:%d:%s\n", file, line, func);
	printf("[DSCR] %s\n", dscr);
}

void assert_float(float v1 , float v2, const char *dscr, const char * func, const char *file, int line)
{
	if (v1 != v2)
	{
		printf("[FAIL] %s:%d:%s\n", file, line, func);
		printf("[INFO] %s:%f != %s:%f\n", getname(v1), v1, getname(v2),v2);
		printf("[DSCR] %s\n", dscr);
		return ;
	}
	printf("[PASS] %s:%d:%s\n", file, line, func);
	printf("[DSCR] %s\n", dscr);
}

void assert_int(int v1 , int v2, const char *dscr, const char * func, const char *file, int line)
{
	if (v1 != v2)
	{
		printf("[FAIL] %s:%d:%s\n", file, line, func);
		printf("[INFO] %s:%d != %s:%d\n", getname(v1), v1, getname(v2),v2);
		printf("[DSCR] %s\n", dscr);
		return ;
	}
	printf("[PASS] %s:%d:%s\n", file, line, func);
	printf("[DSCR] %s\n", dscr);
}

void assert_complex_double(double complex v1 , double complex v2, const char *dscr, const char * func, const char *file, int line)
{
	if (v1 != v2)
	{
		printf("[FAIL] %s:%d:%s\n", file, line, func);
		printf("[INFO] %s:%f+%f*I != %s:%f+%f*I\n", getname(v1), creal(v1), cimag(v1), getname(v2),creal(v2), cimag(v2));
		printf("[DSCR] %s\n", dscr);
		return ;
	}
	printf("[PASS] %s:%d:%s\n", file, line, func);
	printf("[DSCR] %s\n", dscr);
}
void assert_ns(float v1 , float v2, const char *dscr, const char * func, const char *file, int line)
{

	printf("[FAIL] %s:%d:%s\n", file, line, func);
	printf("[INFO] not supported type\n");
	printf("[DSCR] %s\n", dscr);
	return ;
}

int	test_indexing_int(void)
{
	int m_1[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6, 8, 8, 0, 45, 0, 1, 0, 0, 1, 200, 33, 5, 57, 62, 70, 103, 141, 122, 26, 36, 82, 8, 10, 4115, 22, 1, 11, 1, 19};
	int m_1_shape[] = {5, 8};
	t_ndarray *x;
	int index;
	int c_index;;
	int value;
	int c_value;;

	x = init_array((char *)m_1, 2, m_1_shape, sizeof(int));
	// testing the index [3, 2]
	index = 3 * x->strides[0] + 2 * x->strides[1];
	c_index = 26;
	m_assert(index , c_index, "testing the strides"); //testing the strides
	m_assert(get_index(x, 3, 2) , c_index, "testing the indexing function"); //testing the indexing function
	// testing the value with the index [0, 0]
	value = x->data->int_nd[index];
	c_value = 103;
	m_assert(value , c_value, "testing the value");

	return (1);
}

int	test_indexing_double(void)
{
	double m_1[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 8, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1, 200, 33, 5, 57, 62, 70, 103.009, 141, 122, 26.50, 36.334, 82, 8.44002, 10.056, 4115, 22.1, 1.1102, 011.25, 1.01110005, 19};
	int m_1_shape[] = {5, 8};
	t_ndarray *x;
	int index;
	int c_index;
	double value;
	double c_value;

	x = init_array((char *)m_1, 2, m_1_shape, sizeof(double));
	// testing the index [3, 2]
	index = 3 * x->strides[0] + 2 * x->strides[1];
	c_index = 26;
	m_assert(index , c_index, "testing the strides"); //testing the strides
	m_assert(get_index(x, 3, 2) , c_index, "testing the indexing function"); //testing the indexing function
	// testing the value with the index [0, 0]
	value = x->data->double_nd[index];
	c_value = 103.009;
	m_assert(value , c_value, "testing the value");

	return (1);
}

int	test_indexing_complex_double(void)
{
	double complex m_1[] = {0.37 + 0.588*I,  0.92689451+0.57106791*I, 0.93598206+0.30289964*I,  0.54404246+0.09516331*I, 0.02827254+0.00432899*I,  0.06873651+0.24810741*I, 0.94040543+0.43508215*I,  0.58532094+0.67890618*I, 0.68742283+0.64951155*I,  0.15372315+0.89699101*I};
	int m_1_shape[] = {5, 2};
	t_ndarray *x;
	int index;
	int c_index;;
	double complex value;
	double complex c_value;;

	x = init_array((char *)m_1, 2, m_1_shape, sizeof(double complex));
	// testing the index [3, 1]
	index = 3 * x->strides[0] + 1 * x->strides[1];
	c_index = 7;
	m_assert(index , c_index, "testing the strides"); //testing the strides
	m_assert(get_index(x, 3, 1) , c_index, "testing the indexing function"); //testing the indexing function
	// testing the value with the index [0, 0]
	value = x->data->complex_double[index];
	c_value = 0.58532094+0.67890618*I;
	m_assert(value , c_value, "testing the value");

	return (1);
}

/* 
**	slicing tests
*/

int	test_slicing_int(void)
{
	int m_1[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6, 8, 8, 0, 45, 0, 1, 0, 0, 1, 200, 33, 5, 57, 62, 70, 103, 141, 122, 26, 36, 82, 8, 10, 4115, 22, 1, 11, 1, 19};
	int m_1_shape[] = {8, 5};
	t_ndarray *x;
	t_ndarray *slice;
	int index;
	int c_index;;
	int value;
	int c_value;;

	x = init_array((char *)m_1, 2, m_1_shape, sizeof(int));

	slice = make_slice(x, slice_data(1, 2, 1), slice_data(0, 5, 2), NULL);
	
	c_index = 5;
	for (int i = 0; i < slice->shape[0]; i++)
	{
		for (int j = 0; j < slice->shape[1]; j++)
		{
			value = slice->data->int_nd[get_index(slice, i, j)];
			c_value = m_1[c_index];
			c_index+=2;
            if (value != c_value)
			    m_assert(value , c_value, "testing slice values");
		}
	}

	c_value = 1337;
	slice->data->int_nd[get_index(slice, 0, 1)] = c_value;
	value = x->data->int_nd[get_index(x, 1, 2)];
	m_assert(value , c_value, "testing slice assiengment");
	return (1);
}

int	test_slicing_double(void)
{
	double m_1[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 8, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1, 200, 33, 5, 57, 62, 70, 103.009, 141, 122, 26.50, 36.334, 82, 8.44002, 10.056, 4115, 22.1, 1.1102, 011.25, 1.01110005, 19};
	int m_1_shape[] = {8, 5};
	t_ndarray *x;
	t_ndarray *slice;
	int index;
	int c_index;;
	double value;
	double c_value;;

	x = init_array((char *)m_1, 2, m_1_shape, sizeof(double));

	slice = make_slice(x, slice_data(1, 2, 1), slice_data(0, 5, 2), NULL);
	
	c_index = 5;
	for (int i = 0; i < slice->shape[0]; i++)
	{
		for (int j = 0; j < slice->shape[1]; j++)
		{
			value = slice->data->double_nd[get_index(slice, i, j)];
			c_value = m_1[c_index];
			c_index+=2;
            if (value != c_value) // to not spam the test because of the loop
			    m_assert(value , c_value, "testing slice values");
		}
	}

	c_value = 0.1337;
	slice->data->double_nd[get_index(slice, 0, 1)] = c_value;
	value = x->data->double_nd[get_index(x, 1, 2)];
	m_assert(value, c_value, "testing slice assiengment");
	return (1);
}

int	test_slicing_complex_double(void)
{
	double complex m_1[] = {0.37 + 0.588*I,  0.92689451+0.57106791*I, 0.93598206+0.30289964*I,  0.54404246+0.09516331*I, 0.02827254+0.00432899*I,
			  0.06873651+0.24810741*I, 0.94040543+0.43508215*I,  0.58532094+0.67890618*I, 0.68742283+0.64951155*I,  0.15372315+0.89699101*I};
	int m_1_shape[] = {2, 5};
	t_ndarray *x;
	t_ndarray *slice;
	int index;
	int c_index;;
	double complex value;
	double complex c_value;;

	x = init_array((char *)m_1, 2, m_1_shape, sizeof(double complex));

	slice = make_slice(x, slice_data(1, 2, 1), slice_data(0, 5, 2), NULL);
	
	c_index = 5;
	for (int i = 0; i < slice->shape[0]; i++)
	{
		for (int j = 0; j < slice->shape[1]; j++)
		{
			value = slice->data->complex_double[get_index(slice, i, j)];
			c_value = m_1[c_index];
			c_index+=2;
            if (value != c_value) // to not spam the test because of the loop
			    m_assert(value , c_value, "testing slice values");
		}
	}

	c_value = 0.13 + 0.37*I;
	slice->data->complex_double[get_index(slice, 0, 1)] = c_value;
	value = x->data->complex_double[get_index(x, 1, 2)];
	m_assert(value, c_value, "testing slice assiengment");
	return (1);
}

int main(void)
{
	/* indexing tests */
	test_indexing_double();
	test_indexing_int();
	test_indexing_complex_double();

	/* slicing tests */
	test_slicing_double();
	test_slicing_int();
	test_slicing_complex_double();

	write(1, "ALL the tests PASSed.\n", 22);
	return (0);
}
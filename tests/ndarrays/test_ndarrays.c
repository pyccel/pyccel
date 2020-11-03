#include "ndarrays.h"
#include <unistd.h>


#define getname(X) #X
#define my_assert(X , Y, dscr) _Generic((X), double: assert_double,\
                            float: assert_float,\
                            int: assert_int,\
                            double complex : assert_cdouble,\
                            default: assert_ns)(X , Y, getname(X), getname(Y), dscr, __func__, __FILE__, __LINE__)

void assert_double(double v1 , double v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] %s:%f != %s:%f\n", v1_name, v1, v2_name,v2);
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n", file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void assert_float(float v1 , float v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] %s:%f != %s:%f\n", v1_name, v1, v2_name,v2);
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n", file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void assert_int(int v1 , int v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] %s:%d != %s:%d\n", v1_name, v1, v2_name,v2);
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n", file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void assert_cdouble(double complex v1 , double complex v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] %s:%f+%f*I != %s:%f+%f*I\n", v1_name, creal(v1), cimag(v1), v2_name,creal(v2), cimag(v2));
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n", file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void assert_ns(float v1 , float v2,
        const char *v1_name, const char *v2_name, const char *dscr,
        const char * func, const char *file, int line)
{
    printf("[FAIL] %s:%d:%s\n", file, line, func);
    printf("[INFO] not supported type\n");
    printf("[DSCR] %s\n", dscr);
}

int test_indexing_int(void)
{
    int m_1[] = {2, 3, 5, 5, 6, 7, 10, 11,
                12, 260, 6, 8, 8, 0, 45, 0,
                1, 0, 0, 1, 200, 33, 5, 57,
                62, 70, 103, 141, 122, 26, 36, 82,
                8, 10, 4115, 22, 1, 11, 1, 19};
    int m_1_shape[] = {5, 8};
    t_ndarray x;
    int index;
    int c_index;
    int value;
    int c_value;

    x = array_create(2, m_1_shape, nd_int);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 26;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_int[index];
    c_value = 103;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int test_indexing_double(void)
{
    double m_1[] = {2, 3, 5, 5, 6, 7, 10, 11,
                    12, 260, 6.34, 8, 8.002, 0.056, 45, 0.1,
                    1.02, 0.25, 0.00005, 1, 200, 33, 5, 57,
                    62, 70, 103.009, 141, 122, 26.50, 36.334, 82,
                    8.44002, 10.056, 4115, 22.1, 1.1102, 011.25, 1.01110005, 19};
    int m_1_shape[] = {5, 8};
    t_ndarray x;
    int index;
    int c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 26;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_double[index];
    c_value = 103.009;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int test_indexing_cdouble(void)
{
    double complex m_1[] = {0.37 + 0.588*I,  0.92689451+0.57106791*I,
                            0.93598206+0.30289964*I,  0.54404246+0.09516331*I,
                            0.02827254+0.00432899*I,  0.06873651+0.24810741*I,
                            0.94040543+0.43508215*I,  0.58532094+0.67890618*I,
                            0.68742283+0.64951155*I,  0.15372315+0.89699101*I};
    int m_1_shape[] = {5, 2};
    t_ndarray x;
    int index;
    int c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_cdouble[index];
    c_value = 0.58532094+0.67890618*I;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}


/* 
**  slicing tests
*/

int test_slicing_int(void)
{
    int m_1[] = {2, 3, 5, 5, 6,
                7, 10, 11, 12, 260,
                6, 8, 8, 0, 45,
                0, 1, 0, 0, 1,
                200, 33, 5, 57, 62,
                70, 103, 141, 122, 26,
                36, 82, 8, 10, 4115,
                22, 1, 11, 1, 19};
    int m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray slice;
    int c_index;
    int value;
    int c_value;

    x = array_create(2, m_1_shape, nd_int);
    memcpy(x.raw_data, m_1, x.buffer_size);
    slice = array_slicing(x, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 5;
    for (int i = 0; i < slice.shape[0]; i++)
    {
        for (int j = 0; j < slice.shape[1]; j++)
        {
            value = slice.nd_int[get_index(slice, i, j)];
            c_value = m_1[c_index];
            c_index+=2;
            if (value != c_value)
                my_assert(value , c_value, "testing slice values");
        }
    }
    c_value = 1337;
    slice.nd_int[get_index(slice, 0, 1)] = c_value;
    value = x.nd_int[get_index(x, 1, 2)];
    my_assert(value , c_value, "testing slice assignment");
    free_array(x);
    free_array(slice);
    return (0);
}

int test_slicing_double(void)
{
    double m_1[] = {2, 3, 5, 5, 6,
                    7, 10, 11, 12, 260,
                    6.34, 8, 8.002, 0.056, 45,
                    0.1, 1.02, 0.25, 0.00005, 1,
                    200, 33, 5, 57, 62,
                    103.009, 141, 122, 26.50, 36.334,
                    82, 8.44002, 10.056, 4115, 22.1,
                    1.1102, 011.25, 1.01110005, 19, 70};
    int m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray slice;
    int c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double);
    memcpy(x.raw_data, m_1, x.buffer_size);
    slice = array_slicing(x, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 5;
    for (int i = 0; i < slice.shape[0]; i++)
    {
        for (int j = 0; j < slice.shape[1]; j++)
        {
            value = slice.nd_double[get_index(slice, i, j)];
            c_value = m_1[c_index];
            c_index+=2;
            if (value != c_value) // to not spam the test because of the loop
                my_assert(value , c_value, "testing slice values");
        }
    }
    c_value = 0.1337;
    slice.nd_double[get_index(slice, 0, 1)] = c_value;
    value = x.nd_double[get_index(x, 1, 2)];
    my_assert(value, c_value, "testing slice assignment");
    free_array(x);
    free_array(slice);
    return (0);
}

int test_slicing_cdouble(void)
{
    double complex m_1[] = {
                    0.37 + 0.588*I,  0.92+0.57*I, 0.93+0.30*I,  0.54+0.09*I, 0.02+0.01*I,
                    0.03+0.24*I, 0.94+0.43*I,  0.58+0.67*I, 0.68+0.64*I,  0.15+0.89*I
                    };
    int m_1_shape[] = {2, 5};
    t_ndarray x;
    t_ndarray slice;
    int c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble);
    memcpy(x.raw_data, m_1, x.buffer_size);
    slice = array_slicing(x, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 5;
    for (int i = 0; i < slice.shape[0]; i++)
    {
        for (int j = 0; j < slice.shape[1]; j++)
        {
            value = slice.nd_cdouble[get_index(slice, i, j)];
            c_value = m_1[c_index];
            c_index+=2;
            if (value != c_value) // to not spam the test because of the loop
                my_assert(value , c_value, "testing slice values");
        }
    }
    c_value = 0.13 + 0.37*I;
    slice.nd_cdouble[get_index(slice, 0, 1)] = c_value;
    value = x.nd_cdouble[get_index(x, 1, 2)];
    my_assert(value, c_value, "testing slice assignment");
    free_array(x);
    free_array(slice);
    return (0);
}

/* array_fill tests */

int test_array_fill_int(void)
{
    int m_1_shape[] = {5, 2};
    t_ndarray x;
    int index;
    int c_index;
    int value;
    int c_value;

    x = array_create(2, m_1_shape, nd_int);
    array_fill(32, x);
    // testing the index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int[index];
    c_value = 32;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int test_array_fill_double(void)
{
    int m_1_shape[] = {5, 2};
    t_ndarray x;
    int index;
    int c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double);
    array_fill(2., x);
    // testing the index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_double[index];
    c_value = 2.;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int test_array_fill_cdouble(void)
{
    int m_1_shape[] = {5, 2};
    t_ndarray x;
    int index;
    int c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble);
    array_fill(0.3+0.54*I, x);
    // testing the index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_cdouble[index];
    c_value = 0.3+0.54*I;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

/* array_zeros tests */

int test_array_zeros_double(void)
{
    int m_1_shape[] = {5, 2};
    t_ndarray x;
    int index;
    int c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double);
    array_fill(0, x);
    // testing the index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_double[index];
    c_value = 0.;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int test_array_zeros_int(void)
{
    int m_1_shape[] = {5, 2};
    t_ndarray x;
    int index;
    int c_index;
    int value;
    int c_value;

    x = array_create(2, m_1_shape, nd_int);
    array_fill(0, x);
    // testing the index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int[index];
    c_value = 0;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int test_array_zeros_cdouble(void)
{
    int m_1_shape[] = {5, 2};
    t_ndarray x;
    int index;
    int c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble);
    array_fill(0, x);
    // testing the index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_cdouble[index];
    c_value = 0+0*I;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int main(void)
{
    /* indexing tests */
    test_indexing_double();
    test_indexing_int();
    test_indexing_cdouble();
    /* slicing tests */
    test_slicing_double();
    test_slicing_int();
    test_slicing_cdouble();
    /* array_fill tests */
    test_array_fill_int();
    test_array_fill_double();
    test_array_fill_cdouble();
    /* array_zeros tests */
    test_array_zeros_int();
    test_array_zeros_double();
    test_array_zeros_cdouble();
    return (0);
}

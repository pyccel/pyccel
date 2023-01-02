/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#include "ndarrays.h"
#include <unistd.h>
#include <stdio.h>
#include <string.h>


#define getname(X) #X
#define my_assert(X , Y, dscr) _Generic((X), double: assert_double,\
                            float: assert_float,\
                            int64_t: assert_int64,\
                            int32_t: assert_int32,\
                            int16_t: assert_int16,\
                            int8_t: assert_int8,\
                            float complex : assert_cfloat,\
                            double complex : assert_cdouble,\
                            default: assert_ns)(X , Y, getname(X), getname(Y), dscr, __func__, __FILE__, __LINE__)

void assert_double(double v1 , double v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int32_t line)
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
        const char * func, const char *file, int32_t line)
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

void assert_int64(int64_t v1 , int64_t v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int32_t line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] %s:%ld != %s:%ld\n", v1_name, v1, v2_name,v2);
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n", file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void assert_int32(int32_t v1 , int32_t v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int32_t line)
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

void assert_int16(int16_t v1 , int16_t v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int32_t line)
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

void assert_int8(int8_t v1 , int8_t v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int32_t line)
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

void assert_cfloat(float complex v1 , float complex v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int32_t line)
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

void assert_cdouble(double complex v1 , double complex v2,
        const char *v1_name, const char *v2_name,const char *dscr,
        const char * func, const char *file, int32_t line)
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
        const char * func, const char *file, int32_t line)
{
    printf("[FAIL] %s:%d:%s\n", file, line, func);
    printf("[INFO] not supported type\n");
    printf("[DSCR] %s\n", dscr);
}

int32_t test_indexing_int64(void)
{
    int64_t m_1[] = {2, 3, 5, 5, 6, 7, 10, 11,
                12, 260, 6, 8, 8, 0, 45, 0,
                1, 0, 0, 1, 200, 33, 5, 57,
                62, 70, 103, 141, 122, 26, 36, 82,
                8, 10, 4115, 22, 1, 11, 1, 19};
    int64_t m_1_shape[] = {5, 8};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int64_t value;
    int64_t c_value;

    x = array_create(2, m_1_shape, nd_int64, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 26;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_int64[index];
    c_value = 103;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_indexing_int64_order_f(void)
{
    int64_t m_1[] = {1, 6, 6, 2, 4, 0, 1, 5,
                    7, 1, 3, 5, 3, 6, 4, 9,
                    7, 6, 7, 1, 7, 7, 4, 4,
                    7, 1, 9, 0, 1, 2, 4, 5,
                    6, 1, 7, 3, 1, 5, 9, 6};
    int64_t m_1_shape[] = {5, 8};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int64_t value;
    int64_t c_value;

    x = array_create(2, m_1_shape, nd_int64, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 13;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_int64[index];
    c_value = 6;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_indexing_int32(void)
{
    int32_t m_1[] = {2, 3, 5, 5, 6, 7, 10, 11,
                12, 260, 6, 8, 8, 0, 45, 0,
                1, 0, 0, 1, 200, 33, 5, 57,
                62, 70, 103, 141, 122, 26, 36, 82,
                8, 10, 4115, 22, 1, 11, 1, 19};
    int64_t m_1_shape[] = {5, 8};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int32_t value;
    int32_t c_value;

    x = array_create(2, m_1_shape, nd_int32, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 26;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_int32[index];
    c_value = 103;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_indexing_int32_order_f(void)
{
    int32_t m_1[] = {1, 6, 6, 2, 4, 0, 1, 5,
                    7, 1, 3, 5, 3, 6, 4, 9,
                    7, 6, 7, 1, 7, 7, 4, 4,
                    7, 1, 9, 0, 1, 2, 4, 5,
                    6, 1, 7, 3, 1, 5, 9, 6};
    int64_t m_1_shape[] = {5, 8};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int32_t value;
    int32_t c_value;

    x = array_create(2, m_1_shape, nd_int32, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 13;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_int32[index];
    c_value = 6;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}


int32_t test_indexing_int16(void)
{
    int16_t m_1[] = {2, 3, 5, 5, 6, 7, 10, 11,
                12, 260, 6, 8, 8, 0, 45, 0,
                1, 0, 0, 1, 200, 33, 5, 57,
                62, 70, 103, 141, 122, 26, 36, 82,
                8, 10, 4115, 22, 1, 11, 1, 19};
    int64_t m_1_shape[] = {5, 8};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int16_t value;
    int16_t c_value;

    x = array_create(2, m_1_shape, nd_int16, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 26;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_int16[index];
    c_value = 103;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_indexing_int16_order_f(void)
{
    int16_t m_1[] = {1, 6, 6, 2, 4, 0, 1, 5,
                    7, 1, 3, 5, 3, 6, 4, 9,
                    7, 6, 7, 1, 7, 7, 4, 4,
                    7, 1, 9, 0, 1, 2, 4, 5,
                    6, 1, 7, 3, 1, 5, 9, 6};
    int64_t m_1_shape[] = {5, 8};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int16_t value;
    int16_t c_value;

    x = array_create(2, m_1_shape, nd_int16, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 13;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_int16[index];
    c_value = 6;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_indexing_int8(void)
{
    int8_t m_1[] = {2, 3, 5, 5, 6, 7, 10, 11,
                12, 250, 6, 8, 8, 0, 45, 0,
                1, 0, 0, 1, 200, 33, 5, 57,
                62, 70, 103, 141, 122, 26, 36, 82,
                8, 10, 251, 22, 1, 11, 1, 19};
    int64_t m_1_shape[] = {5, 8};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int8_t value;
    int8_t c_value;

    x = array_create(2, m_1_shape, nd_int8, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 26;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_int8[index];
    c_value = 103;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_indexing_int8_order_f(void)
{
    int8_t m_1[] = {1, 6, 6, 2, 4, 0, 1, 5,
                    7, 1, 3, 5, 3, 6, 4, 9,
                    7, 6, 7, 1, 7, 7, 4, 4,
                    7, 1, 9, 0, 1, 2, 4, 5,
                    6, 1, 7, 3, 1, 5, 9, 6};
    int64_t m_1_shape[] = {5, 8};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int8_t value;
    int8_t c_value;

    x = array_create(2, m_1_shape, nd_int8, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 13;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_int8[index];
    c_value = 6;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_indexing_double(void)
{
    double m_1[] = {2, 3, 5, 5, 6, 7, 10, 11,
                    12, 260, 6.34, 8, 8.002, 0.056, 45, 0.1,
                    1.02, 0.25, 0.00005, 1, 200, 33, 5, 57,
                    62, 70, 103.009, 141, 122, 26.50, 36.334, 82,
                    8.44002, 10.056, 4115, 22.1, 1.1102, 011.25, 1.01110005, 19};
    int64_t m_1_shape[] = {5, 8};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 2]
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

int32_t test_indexing_double_order_f(void)
{
    double m_1[] = {0.05751476201087158, 0.13745300160998386, 0.2079973756146688, 0.2141008701292113,
                        0.2841626347062822, 0.4239597742653679, 0.6522693509949357, 0.9230925248474112,
                    0.8525960970856643, 0.8138267834551158, 0.5210862950661376, 0.5971376449046766,
                        0.21489965384896903, 0.39310326602036905, 0.8762355339594128, 0.6760453456313423,
                    0.9424121921037278, 0.6475903689621096, 0.4663033100019591, 0.9317847307880955,
                        0.6932017388898739, 0.2543468648524083, 0.6020488047717303, 0.5951783988465483,
                    0.45982787529445157, 0.6810379211895127, 0.6125792953364687, 0.7803671441186834,
                        0.24092775530531474, 0.3129576055158103, 0.6417260363979195, 0.029619310490432538,
                    0.04414839864241349, 0.9515999281039909, 0.2694450127647845, 0.35691412053397653,
                        0.790349738990523, 0.23743626350675473, 0.5085166254064067, 0.06735188687983484};
    int64_t m_1_shape[] = {5, 8};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 2]
    index = 3 * x.strides[0] + 2 * x.strides[1];
    c_index = 13;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 2) , c_index, "testing the indexing function");
    // testing the value with the index [3, 2]
    value = x.nd_double[index];
    c_value = 0.39310326602036905;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_indexing_cdouble(void)
{
    double complex m_1[] = {0.37 + 0.588*I,  0.92689451+0.57106791*I,
                            0.93598206+0.30289964*I,  0.54404246+0.09516331*I,
                            0.02827254+0.00432899*I,  0.06873651+0.24810741*I,
                            0.94040543+0.43508215*I,  0.58532094+0.67890618*I,
                            0.68742283+0.64951155*I,  0.15372315+0.89699101*I};
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 1]
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

int32_t test_indexing_cdouble_order_f(void)
{
    double complex m_1[] = {(0.8399107+0.39774981*I), (0.11668451+0.44048746*I),
                            (0.82147723+0.65709434*I), (0.19273834+0.24206348*I),
                            (0.88065081+0.34479325*I), (0.17994627+0.03374548*I),
                            (0.22043986+0.16516544*I), (0.03286256+0.1403558*I),
                            (0.02005023+0.59950758*I), (0.96582983+0.89097352*I)};
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 8;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_cdouble[index];
    c_value = 0.02005023+0.59950758*I;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_indexing_cfloat(void)
{
    float complex m_1[] = {0.37 + 0.588*I,  0.92689451+0.57106791*I,
                            0.93598206+0.30289964*I,  0.54404246+0.09516331*I,
                            0.02827254+0.00432899*I,  0.06873651+0.24810741*I,
                            0.94040543+0.43508215*I,  0.58532094+0.67890618*I,
                            0.68742283+0.64951155*I,  0.15372315+0.89699101*I};
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    float complex value;
    float complex c_value;

    x = array_create(2, m_1_shape, nd_cfloat, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_cfloat[index];
    c_value = 0.58532094+0.67890618*I;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_indexing_cfloat_order_f(void)
{
    float complex m_1[] = {(0.7531014552678067+0.3212445034096577*I), (0.9124849621552715+0.5009509557101085*I),
                            (0.516709098068522+0.8742994690327561*I), (0.8641458676501654+0.1568974703392113*I),
                            (0.6840573468633404+0.7427309400709274*I), (0.8727371859517135+0.19214143697029318*I),
                            (0.8299801130099297+0.476220186802197*I), (0.9218646531661248+0.8437506286920766*I),
                            (0.8023905547293535+0.17882460280193202*I), (0.8096431613369838+0.7072085622315805*I)};
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    float complex value;
    float complex c_value;

    x = array_create(2, m_1_shape, nd_cfloat, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 8;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_cfloat[index];
    c_value = 0.8023905547293535+0.17882460280193202*I;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

/*
**  slicing tests
*/

int32_t test_slicing_int64(void)
{
    int64_t m_1[] = {2, 3, 5, 5, 6,
                7, 10, 11, 12, 260,
                6, 8, 8, 0, 45,
                0, 1, 0, 0, 1,
                200, 33, 5, 57, 62,
                70, 103, 141, 122, 26,
                36, 82, 8, 10, 4115,
                22, 1, 11, 1, 19};
    int64_t m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    int64_t value;
    int64_t c_value;

    x = array_create(2, m_1_shape, nd_int64, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 5;
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_int64[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=2;
            if (value != c_value)
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 1337;
    xview.nd_int64[get_index(xview, 0, 1)] = c_value;
    value = x.nd_int64[get_index(x, 1, 2)];
    my_assert(value , c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_int64_order_f(void)
{
    int64_t m_1[] = {0, 5, 10, 15, 20, 25, 30, 35,
                    1, 6, 11, 16, 21, 26, 31, 36,
                    2, 7, 12, 17, 22, 27, 32, 37,
                    3, 8, 13, 18, 23, 28, 33, 38,
                    4, 9, 14, 19, 24, 29, 34, 39};
    int64_t m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    int64_t value;
    int64_t c_value;

    x = array_create(2, m_1_shape, nd_int64, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2)); // [5, 7, 9] is the resulting view
    c_index = 1; 
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_int64[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=16;
            if (value != c_value)
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 1337;
    xview.nd_int64[get_index(xview, 0, 1)] = c_value;
    value = x.nd_int64[get_index(x, 1, 2)];
    my_assert(value , c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_int32(void)
{
    int32_t m_1[] = {2, 3, 5, 5, 6,
                7, 10, 11, 12, 260,
                6, 8, 8, 0, 45,
                0, 1, 0, 0, 1,
                200, 33, 5, 57, 62,
                70, 103, 141, 122, 26,
                36, 82, 8, 10, 4115,
                22, 1, 11, 1, 19};
    int64_t m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    int32_t value;
    int32_t c_value;

    x = array_create(2, m_1_shape, nd_int32, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 5;
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_int32[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=2;
            if (value != c_value)
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 1337;
    xview.nd_int32[get_index(xview, 0, 1)] = c_value;
    value = x.nd_int32[get_index(x, 1, 2)];
    my_assert(value , c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_int32_order_f(void)
{
    int32_t m_1[] = {0, 5, 10, 15, 20, 25, 30, 35,
                    1, 6, 11, 16, 21, 26, 31, 36,
                    2, 7, 12, 17, 22, 27, 32, 37,
                    3, 8, 13, 18, 23, 28, 33, 38,
                    4, 9, 14, 19, 24, 29, 34, 39};
    int64_t m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    int32_t value;
    int32_t c_value;

    x = array_create(2, m_1_shape, nd_int32, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 1; 
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_int32[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=16;
            if (value != c_value)
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 1337;
    xview.nd_int32[get_index(xview, 0, 1)] = c_value;
    value = x.nd_int32[get_index(x, 1, 2)];
    my_assert(value , c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_int16(void)
{
    int16_t m_1[] = {2, 3, 5, 5, 6,
                7, 10, 11, 12, 260,
                6, 8, 8, 0, 45,
                0, 1, 0, 0, 1,
                200, 33, 5, 57, 62,
                70, 103, 141, 122, 26,
                36, 82, 8, 10, 4115,
                22, 1, 11, 1, 19};
    int64_t m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    int16_t value;
    int16_t c_value;

    x = array_create(2, m_1_shape, nd_int16, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 5;
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_int16[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=2;
            if (value != c_value)
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 1337;
    xview.nd_int16[get_index(xview, 0, 1)] = c_value;
    value = x.nd_int16[get_index(x, 1, 2)];
    my_assert(value , c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_int16_order_f(void)
{
    int16_t m_1[] = {0, 5, 10, 15, 20, 25, 30, 35,
                    1, 6, 11, 16, 21, 26, 31, 36,
                    2, 7, 12, 17, 22, 27, 32, 37,
                    3, 8, 13, 18, 23, 28, 33, 38,
                    4, 9, 14, 19, 24, 29, 34, 39};
    int64_t m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    int16_t value;
    int16_t c_value;

    x = array_create(2, m_1_shape, nd_int16, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 1; 
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_int16[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=16;
            if (value != c_value)
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 1337;
    xview.nd_int16[get_index(xview, 0, 1)] = c_value;
    value = x.nd_int16[get_index(x, 1, 2)];
    my_assert(value , c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_int8(void)
{
    int8_t m_1[] = {2, 3, 5, 5, 6,
                7, 10, 11, 12, 250,
                6, 8, 8, 0, 45,
                0, 1, 0, 0, 1,
                200, 33, 5, 57, 62,
                70, 103, 141, 122, 26,
                36, 82, 8, 10, 251,
                22, 1, 11, 1, 19};
    int64_t m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    int8_t value;
    int8_t c_value;

    x = array_create(2, m_1_shape, nd_int8, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 5;
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_int8[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=2;
            if (value != c_value)
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 133;
    xview.nd_int8[get_index(xview, 0, 1)] = c_value;
    value = x.nd_int8[get_index(x, 1, 2)];
    my_assert(value , c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_int8_order_f(void)
{
    int8_t m_1[] = {0, 5, 10, 15, 20, 25, 30, 35,
                    1, 6, 11, 16, 21, 26, 31, 36,
                    2, 7, 12, 17, 22, 27, 32, 37,
                    3, 8, 13, 18, 23, 28, 33, 38,
                    4, 9, 14, 19, 24, 29, 34, 39};
    int64_t m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    int8_t value;
    int8_t c_value;

    x = array_create(2, m_1_shape, nd_int8, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 1; 
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_int8[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=16;
            if (value != c_value)
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 254;
    xview.nd_int8[get_index(xview, 0, 1)] = c_value;
    value = x.nd_int8[get_index(x, 1, 2)];
    my_assert(value , c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_double(void)
{
    double m_1[] = {2, 3, 5, 5, 6,
                    7, 10, 11, 12, 260,
                    6.34, 8, 8.002, 0.056, 45,
                    0.1, 1.02, 0.25, 0.00005, 1,
                    200, 33, 5, 57, 62,
                    103.009, 141, 122, 26.50, 36.334,
                    82, 8.44002, 10.056, 4115, 22.1,
                    1.1102, 011.25, 1.01110005, 19, 70};
    int64_t m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 5;
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_double[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=2;
            if (value != c_value) // to not spam the test because of the loop
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 0.1337;
    xview.nd_double[get_index(xview, 0, 1)] = c_value;
    value = x.nd_double[get_index(x, 1, 2)];
    my_assert(value, c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_double_order_f(void)
{
    double m_1[] = {.0, .5, .10, .15, .20, .25, .30, .35,
                    .1, .6, .11, .16, .21, .26, .31, .36,
                    .2, .7, .12, .17, .22, .27, .32, .37,
                    .3, .8, .13, .18, .23, .28, .33, .38,
                    .4, .9, .14, .19, .24, .29, .34, .39};
    int64_t m_1_shape[] = {8, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 1;
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_double[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=16;
            if (value != c_value) // to not spam the test because of the loop
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 0.1337;
    xview.nd_double[get_index(xview, 0, 1)] = c_value;
    value = x.nd_double[get_index(x, 1, 2)];
    my_assert(value, c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_cdouble(void)
{
    double complex m_1[] = {
                    0.37 + 0.588*I,  0.92+0.57*I, 0.93+0.30*I,  0.54+0.09*I, 0.02+0.01*I,
                    0.03+0.24*I, 0.94+0.43*I,  0.58+0.67*I, 0.68+0.64*I,  0.15+0.89*I
                    };
    int64_t m_1_shape[] = {2, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble, false, order_c);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 5;
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_cdouble[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=2;
            if (value != c_value) // to not spam the test because of the loop
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 0.13 + 0.37*I;
    xview.nd_cdouble[get_index(xview, 0, 1)] = c_value;
    value = x.nd_cdouble[get_index(x, 1, 2)];
    my_assert(value, c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

int32_t test_slicing_cdouble_order_f(void)
{
    double complex m_1[] = {(0.01155429742163272+0.8434047534139181*I), (0.6708962942795746+0.7899918807430825*I),
                            (0.03753686440380166+0.8122524072995655*I), (0.16325109115080305+0.5483841122139524*I),
                            (0.1464506288395122+0.9068262955220873*I), (0.5124547225966671+0.5843860899075526*I),
                            (0.758038809105398+0.6062573686379302*I), (0.29394537852759395+0.24406369358911972*I),
                            (0.4819055755095919+0.9815668938802014*I), (0.3907440258926417+0.5648389124624602*I)};

    int64_t m_1_shape[] = {2, 5};
    t_ndarray x;
    t_ndarray xview;
    int32_t c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble, false, order_f);
    memcpy(x.raw_data, m_1, x.buffer_size);
    xview = array_slicing(x, 2, new_slice(1, 2, 1), new_slice(0, 5, 2));
    c_index = 1;
    for (int32_t i = 0; i < xview.shape[0]; i++)
    {
        for (int32_t j = 0; j < xview.shape[1]; j++)
        {
            value = xview.nd_cdouble[get_index(xview, i, j)];
            c_value = m_1[c_index];
            c_index+=4;
            if (value != c_value) // to not spam the test because of the loop
                my_assert(value , c_value, "testing xview values");
        }
    }
    c_value = 0.13 + 0.37*I;
    xview.nd_cdouble[get_index(xview, 0, 1)] = c_value;
    value = x.nd_cdouble[get_index(x, 1, 2)];
    my_assert(value, c_value, "testing xview assignment");
    free_array(x);
    free_pointer(xview);
    return (0);
}

/* array_fill tests */

int32_t test_array_fill_int64(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int64_t value;
    int64_t c_value;

    x = array_create(2, m_1_shape, nd_int64, false, order_c);
    array_fill((int64_t)32, x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int64[index];
    c_value = 32;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_array_fill_int64_order_f(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int64_t value;
    int64_t c_value;

    x = array_create(2, m_1_shape, nd_int64, false, order_f);
    array_fill((int64_t)64, x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 8;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int64[index];
    c_value = 64;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}


int32_t test_array_fill_int32(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int32_t value;
    int32_t c_value;

    x = array_create(2, m_1_shape, nd_int32, false, order_c);
    array_fill((int32_t)32, x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int32[index];
    c_value = 32;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_array_fill_int32_order_f(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int32_t value;
    int32_t c_value;

    x = array_create(2, m_1_shape, nd_int32, false, order_f);
    array_fill((int32_t)32, x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 8;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int32[index];
    c_value = 32;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}


int32_t test_array_fill_int16(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int16_t value;
    int16_t c_value;

    x = array_create(2, m_1_shape, nd_int16, false, order_c);
    array_fill((int16_t)32, x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int16[index];
    c_value = 32;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_array_fill_int16_order_f(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int16_t value;
    int16_t c_value;

    x = array_create(2, m_1_shape, nd_int16, false, order_f);
    array_fill((int16_t)16, x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 8;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int16[index];
    c_value = 16;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_array_fill_int8(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int8_t value;
    int8_t c_value;

    x = array_create(2, m_1_shape, nd_int8, false, order_c);
    array_fill((int8_t)32, x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int8[index];
    c_value = 32;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_array_fill_int8_order_f(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int8_t value;
    int8_t c_value;

    x = array_create(2, m_1_shape, nd_int8, false, order_f);
    array_fill((int8_t)8, x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 8;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int8[index];
    c_value = 8;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_array_fill_double(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double, false, order_c);
    array_fill(2., x);
    // testing the strides for index [3, 1]
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

int32_t test_array_fill_double_order_f(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double, false, order_f);
    array_fill(2., x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 8;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_double[index];
    c_value = 2.;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_array_fill_cdouble(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble, false, order_c);
    array_fill(0.3+0.54*I, x);
    // testing the strides for index [3, 1]
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

int32_t test_array_fill_cdouble_order_f(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble, false, order_f);
    array_fill(0.3+0.54*I, x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 8;
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

int32_t test_array_zeros_double(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    double value;
    double c_value;

    x = array_create(2, m_1_shape, nd_double, false, order_c);
    array_fill(0, x);
    // testing the strides for index [3, 1]
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

int32_t test_array_zeros_int32(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    int32_t value;
    int32_t c_value;

    x = array_create(2, m_1_shape, nd_int32, false, order_c);
    array_fill(0, x);
    // testing the strides for index [3, 1]
    index = 3 * x.strides[0] + 1 * x.strides[1];
    c_index = 7;
    my_assert(index , c_index, "testing the strides");
    my_assert(get_index(x, 3, 1) , c_index, "testing the indexing function");
    // testing the value with the index [3, 1]
    value = x.nd_int32[index];
    c_value = 0;
    my_assert(value , c_value, "testing the value");
    free_array(x);
    return (0);
}

int32_t test_array_zeros_cdouble(void)
{
    int64_t m_1_shape[] = {5, 2};
    t_ndarray x;
    int32_t index;
    int32_t c_index;
    double complex value;
    double complex c_value;

    x = array_create(2, m_1_shape, nd_cdouble, false, order_c);
    array_fill(0, x);
    // testing the strides for index [3, 1]
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

int32_t main(void)
{
    /* indexing tests */
    test_indexing_double();
    test_indexing_int64();
    test_indexing_int32();
    test_indexing_int16();
    test_indexing_int8();
    test_indexing_cdouble();
    test_indexing_cfloat();
    /* slicing tests */
    test_slicing_double();
    test_slicing_int64();
    test_slicing_int32();
    test_slicing_int16();
    test_slicing_int8();
    test_slicing_double();
    test_slicing_cdouble();
    /* array_fill tests */
    test_array_fill_int64();
    test_array_fill_int32();
    test_array_fill_int16();
    test_array_fill_int8();
    test_array_fill_double();
    test_array_fill_cdouble();
    /* array_zeros tests */
    test_array_zeros_int32();
    test_array_zeros_double();
    test_array_zeros_cdouble();

    // /*************ORDER F**********************/

    /* indexing tests */
    test_indexing_int64_order_f();
    test_indexing_int32_order_f();
    test_indexing_int16_order_f();
    test_indexing_int8_order_f();
    test_indexing_double_order_f();
    test_indexing_cdouble_order_f();
    test_indexing_cfloat_order_f();
    /* slicing tests */
    test_slicing_double_order_f();
    test_slicing_int64_order_f();
    test_slicing_int32_order_f();
    test_slicing_int16_order_f();
    test_slicing_int8_order_f();
    test_slicing_double_order_f();
    test_slicing_cdouble_order_f();
    // /* array_fill tests */
    test_array_fill_int64_order_f();
    test_array_fill_int32_order_f();
    test_array_fill_int16_order_f();
    test_array_fill_int8_order_f();
    test_array_fill_double_order_f();
    test_array_fill_cdouble_order_f();
    return (0);
}

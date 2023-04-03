#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "ndarrays.h"
#include "cuda_ndarrays.h"

void assert_double(double v1 , double v2, const char *dscr,
        const char * func, const char *file, int32_t line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] v1:%f != v2:%f\n", v1, v2);
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n", file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void assert_float(float v1 , float v2, const char *dscr,
        const char * func, const char *file, int32_t line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] v1:%f != v2:%f\n", v1, v2);
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n", file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void assert_int64(int64_t v1, int64_t v2, const char *dscr,
        const char * func, const char *file, int32_t line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] v1:%ld != v2:%ld\n", v1, v2);
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n", file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void assert_int32(int32_t v1 , int32_t v2, const char *dscr,
        const char * func, const char *file, int32_t line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] v1:%d != v2:%d\n", v1, v2);
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n",file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void assert_int16(int16_t v1 , int16_t v2, const char *dscr,
        const char * func, const char *file, int32_t line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] v1:%d != v2:%d\n", v1, v2);
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n", file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void assert_int8(int8_t v1 , int8_t v2, const char *dscr,
        const char * func, const char *file, int32_t line)
{
    if (v1 != v2)
    {
        printf("[FAIL] %s:%d:%s\n", file, line, func);
        printf("[INFO] v1:%d != v2:%d\n", v1, v2);
        printf("[DSCR] %s\n", dscr);
        return ;
    }
    printf("[PASS] %s:%d:%s\n", file, line, func);
    printf("[DSCR] %s\n", dscr);
}

void test_cuda_array_create_host_double()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};

    arr = cuda_array_create(1, tmp_shape, nd_double, false, allocateMemoryOnHost);
    double cuda_array_dummy[] = {1.02, 0.25, 5e-05, 1.0, 200.0, 33.0, 5.0, 57.0, 62.0, 70.0, 103.009, 141.0, 122.0, 26.5};
    cudaMemcpy(arr.nd_double, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_double(arr.nd_double[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free_host(arr);
}

void test_cuda_array_create_managed_double()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_double, false, managedMemory);
    double cuda_array_dummy[] = {1.02, 0.25, 5e-05, 1.0, 200.0, 33.0, 5.0, 57.0, 62.0, 70.0, 103.009, 141.0, 122.0, 26.5};
    cudaMemcpy(arr.nd_double, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_double(arr.nd_double[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
}

void test_cuda_array_create_device_double()
{
    t_ndarray arr = {.shape = NULL};
    t_ndarray b = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_float, false, allocateMemoryOnDevice);
    double cuda_array_dummy[] = {1.02, 0.25, 5e-05, 1.0, 200.0, 33.0, 5.0, 57.0, 62.0, 70.0, 103.009, 141.0, 122.0, 26.5};
    cudaMemcpy(arr.nd_double, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);

    int64_t tmp_shape_0001[] = {INT64_C(14)};
    b = cuda_array_create(1, tmp_shape_0001, nd_double, false, allocateMemoryOnHost);
    cudaMemcpy(b.nd_double, arr.nd_double, b.buffer_size, cudaMemcpyDeviceToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_double(b.nd_double[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
    cuda_free_host(b);
}

void test_cuda_array_create_host_float()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};

    arr = cuda_array_create(1, tmp_shape, nd_float, false, allocateMemoryOnHost);
    float cuda_array_dummy[] = {1.02, 0.25, 5e-05, 1.0, 200.0, 33.0, 5.0, 57.0, 62.0, 70.0, 103.009, 141.0, 122.0, 26.5};
    cudaMemcpy(arr.nd_float, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_float(arr.nd_float[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free_host(arr);
}

void test_cuda_array_create_managed_float()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_float, false, managedMemory);
    float cuda_array_dummy[] = {1.02, 0.25, 5e-05, 1.0, 200.0, 33.0, 5.0, 57.0, 62.0, 70.0, 103.009, 141.0, 122.0, 26.5};
    cudaMemcpy(arr.nd_float, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_float(arr.nd_float[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
}

void test_cuda_array_create_device_float()
{
    t_ndarray arr = {.shape = NULL};
    t_ndarray b = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_float, false, allocateMemoryOnDevice);
    float cuda_array_dummy[] = {1.02, 0.25, 5e-05, 1.0, 200.0, 33.0, 5.0, 57.0, 62.0, 70.0, 103.009, 141.0, 122.0, 26.5};
    cudaMemcpy(arr.nd_float, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);

    int64_t tmp_shape_0001[] = {INT64_C(14)};
    b = cuda_array_create(1, tmp_shape_0001, nd_double, false, allocateMemoryOnHost);
    cudaMemcpy(b.nd_float, arr.nd_float, b.buffer_size, cudaMemcpyDeviceToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_float(b.nd_float[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
    cuda_free_host(b);
}

void test_cuda_array_create_host_int64()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};

    arr = cuda_array_create(1, tmp_shape, nd_int64, false, allocateMemoryOnHost);
    int64_t cuda_array_dummy[] = {1, 0, 0, 1, 200, 33, 5, 57,
                    62, 70, 103, 141, 122, 26};
    cudaMemcpy(arr.nd_int64, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int64(arr.nd_int64[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free_host(arr);
}

void test_cuda_array_create_managed_int64()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_int64, false, managedMemory);
    int64_t cuda_array_dummy[] = {1, 0, 0, 1, 200, 33, 5, 57,
                    62, 70, 103, 141, 122, 26};
    cudaMemcpy(arr.nd_int64, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);


    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int64(arr.nd_int64[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
}

void test_cuda_array_create_device_int64()
{
    t_ndarray arr = {.shape = NULL};
    t_ndarray b = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_int64, false, allocateMemoryOnDevice);
    int64_t cuda_array_dummy[] = {1, 0, 0, 1, 200, 33, 5, 57,
                    62, 70, 103, 141, 122, 26};
    cudaMemcpy(arr.nd_double, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);

    int64_t tmp_shape_0001[] = {INT64_C(14)};
    b = cuda_array_create(1, tmp_shape_0001, nd_int64, false, allocateMemoryOnHost);
    cudaMemcpy(b.nd_int64, arr.nd_int64, b.buffer_size, cudaMemcpyDeviceToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int64(b.nd_int64[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
    cuda_free_host(b);
}


void test_cuda_array_create_host_int32()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};

    arr = cuda_array_create(1, tmp_shape, nd_int32, false, allocateMemoryOnHost);
    int32_t cuda_array_dummy[] = {1, 0, 0, 1, 200, 33, 5, 57,
                    62, 70, 103, 141, 122, 26};
    cudaMemcpy(arr.nd_int32, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int32(arr.nd_int32[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free_host(arr); 
}

void test_cuda_array_create_managed_int32()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_int32, false, managedMemory);
    int32_t cuda_array_dummy[] = {1, 0, 0, 1, 200, 33, 5, 57,
                    62, 70, 103, 141, 122, 26};
    cudaMemcpy(arr.nd_int32, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);


    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int32(arr.nd_int32[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
}

void test_cuda_array_create_device_int32()
{
    t_ndarray arr = {.shape = NULL};
    t_ndarray b = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_int32, false, allocateMemoryOnDevice);
    int32_t cuda_array_dummy[] = {1, 0, 0, 1, 200, 33, 5, 57,
                    62, 70, 103, 141, 122, 26};
    cudaMemcpy(arr.nd_int32, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);

    int64_t tmp_shape_0001[] = {INT64_C(14)};
    b = cuda_array_create(1, tmp_shape_0001, nd_int32, false, allocateMemoryOnHost);
    cudaMemcpy(b.nd_int32, arr.nd_int32, b.buffer_size, cudaMemcpyDeviceToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int32(b.nd_int32[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
    cuda_free_host(b);
}

void test_cuda_array_create_host_int16()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};

    arr = cuda_array_create(1, tmp_shape, nd_int16, false, allocateMemoryOnHost);
    int16_t cuda_array_dummy[] = {1, 0, 0, 1, 200, 33, 5, 57,
                    62, 70, 103, 141, 122, 26};
    cudaMemcpy(arr.nd_int16, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int16(arr.nd_int16[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free_host(arr);
}

void test_cuda_array_create_managed_int16()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_int16, false, managedMemory);
    int16_t cuda_array_dummy[] = {1, 0, 0, 1, 200, 33, 5, 57,
                    62, 70, 103, 141, 122, 26};
    cudaMemcpy(arr.nd_int16, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);


    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int16(arr.nd_int16[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
}

void test_cuda_array_create_device_int16()
{
    t_ndarray arr = {.shape = NULL};
    t_ndarray b = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_int16, false, allocateMemoryOnDevice);
    int16_t cuda_array_dummy[] = {1, 0, 0, 1, 200, 33, 5, 57,
                    62, 70, 103, 141, 122, 26};
    cudaMemcpy(arr.nd_int16, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);

    int64_t tmp_shape_0001[] = {INT64_C(14)};
    b = cuda_array_create(1, tmp_shape_0001, nd_int16, false, allocateMemoryOnHost);
    cudaMemcpy(b.nd_int16, arr.nd_int16, b.buffer_size, cudaMemcpyDeviceToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int16(b.nd_int16[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
    cuda_free_host(b);
}

void test_cuda_array_create_host_int8()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};

    arr = cuda_array_create(1, tmp_shape, nd_int8, false, allocateMemoryOnHost);
    int8_t cuda_array_dummy[] = {1, 0, 0, 1, 116, 33, 5, 57,
                    62, 70, 103, 120, 122, 26};
    cudaMemcpy(arr.nd_int8, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int8(arr.nd_int8[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free_host(arr); 
}

void test_cuda_array_create_managed_int8()
{
    t_ndarray arr = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_int8, false, managedMemory);
    int8_t cuda_array_dummy[] = {1, 0, 0, 1, 116, 33, 5, 57,
                    62, 70, 103, 120, 122, 26};
    cudaMemcpy(arr.nd_int8, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);


    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int8(arr.nd_int8[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
}

void test_cuda_array_create_device_int8()
{
    t_ndarray arr = {.shape = NULL};
    t_ndarray b = {.shape = NULL};

    int64_t tmp_shape[] = {INT64_C(14)};
    arr = cuda_array_create(1, tmp_shape, nd_int8, false, allocateMemoryOnDevice);
    int8_t cuda_array_dummy[] = {1, 0, 0, 1, 116, 33, 5, 57,
                    62, 70, 103, 120, 122, 26};
    cudaMemcpy(arr.nd_int8, cuda_array_dummy, arr.buffer_size, cudaMemcpyHostToDevice);

    int64_t tmp_shape_0001[] = {INT64_C(14)};
    b = cuda_array_create(1, tmp_shape_0001, nd_int8, false, allocateMemoryOnHost);
    cudaMemcpy(b.nd_int8, arr.nd_int8, b.buffer_size, cudaMemcpyDeviceToHost);

    assert_int32(arr.nd, 1, "testing the number of dimensions", __func__, __FILE__, __LINE__);
    assert_int64(arr.shape[0], tmp_shape[0], "testing the shape", __func__, __FILE__, __LINE__);
    for(int i = 0; i < tmp_shape[0]; i++)
    {
        assert_int8(b.nd_int8[i], cuda_array_dummy[i], "testing the data", __func__, __FILE__, __LINE__);
    }
    cuda_free(arr);
    cuda_free_host(b);
}

int32_t main(void)
{
    /* Cuda array creation tests */
    test_cuda_array_create_host_double();
    test_cuda_array_create_managed_double();
    test_cuda_array_create_device_double();

    test_cuda_array_create_host_float();
    test_cuda_array_create_managed_float();
    test_cuda_array_create_device_float();

    test_cuda_array_create_host_int64();
    test_cuda_array_create_managed_int64();
    test_cuda_array_create_device_int64();

    test_cuda_array_create_host_int32();
    test_cuda_array_create_managed_int32();
    test_cuda_array_create_device_int32();

    test_cuda_array_create_host_int16();
    test_cuda_array_create_managed_int16();
    test_cuda_array_create_device_int16();

    test_cuda_array_create_host_int8();
    test_cuda_array_create_managed_int8();
    test_cuda_array_create_device_int8();

    return (0);
}

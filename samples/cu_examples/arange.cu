extern "C"
{
	#include "ndarrays/ndarrays.h"
}
#include <stdio.h>

__global__
void cuda_array_arange(t_ndarray arr, int start)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for(int i = index ; i < arr.length; i+=stride)
		arr.nd_int64[i] = (i + start);
}

t_ndarray   cuda_array_create(int32_t nd, int64_t *shape, enum e_types type)
{
    t_ndarray arr;

    arr.nd = nd;
    arr.type = type;
    switch (type)
    {
        case nd_int8:
            arr.type_size = sizeof(int8_t);
            break;
        case nd_int16:
            arr.type_size = sizeof(int16_t);
            break;
        case nd_int32:
            arr.type_size = sizeof(int32_t);
            break;
        case nd_int64:
            arr.type_size = sizeof(int64_t);
            break;
        case nd_float:
            arr.type_size = sizeof(float);
            break;
        case nd_double:
            arr.type_size = sizeof(double);
            break;
        case nd_bool:
            arr.type_size = sizeof(bool);
            break;
    }
    arr.is_view = false;
    arr.length = 1;
    cudaMallocManaged(&(arr.shape), arr.nd * sizeof(int64_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.length *= shape[i];
        arr.shape[i] = shape[i];
    }
    arr.buffer_size = arr.length * arr.type_size;
    cudaMallocManaged(&(arr.strides), nd * sizeof(int64_t));
    for (int32_t i = 0; i < arr.nd; i++)
    {
        arr.strides[i] = 1;
        for (int32_t j = i + 1; j < arr.nd; j++)
            arr.strides[i] *= arr.shape[j];
    }
    cudaMallocManaged(&(arr.raw_data), arr.buffer_size);
    return (arr);
}

__global__
void	mat_prod(t_ndarray mat_p, t_ndarray mat_1, t_ndarray mat_2)
{
	int i;
	int i_x;
	int i_y;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= mat_p.length)
		return;
	i_x = i / mat_p.strides[0];
	i_y = i % mat_p.strides[0];
	//printf("%d %d %d\n", i , i_x, i_y);
	mat_p.nd_int64[mat_p.strides[0] * i_x + mat_p.strides[1] * i_y] = 0;
	for (int j = 0; j < mat_p.shape[0]; j++)
	{
		//printf("%d %d %d - %ld += %ld * %ld\n", i, i_x, i_y, mat_p.nd_int64[mat_p.strides[0] * i_x + mat_p.strides[1] * i_y], mat_1.nd_int64[mat_1.strides[0] * i_x + mat_1.strides[1] * j], mat_2.nd_int64[mat_2.strides[0] * j + mat_2.strides[1] * i_x]);
		mat_p.nd_int64[mat_p.strides[0] * i_x + mat_p.strides[1] * i_y] +=
			mat_1.nd_int64[mat_1.strides[0] * i_x + mat_1.strides[1] * j]
			* mat_2.nd_int64[mat_2.strides[0] * j + mat_2.strides[1] * i_y];
	}
	//printf("%ld %ld - %ld - %ld\n", i_x * mat_p.strides[0], i_y * mat_p.strides[1], i_x * mat_p.strides[0] + i_y * mat_p.strides[1], mat_p.nd_int64[mat_p.strides[0] * i_x + mat_p.strides[1] * i_y]);
}

int main(void)
{
	t_ndarray mat_1;
	t_ndarray mat_2;
	t_ndarray mat_r;
	int64_t a[] = {1000,1000};
	int block_size;
	int nblocks;

	mat_1 = cuda_array_create(2, a, nd_int64);
	mat_2 = cuda_array_create(2, a, nd_int64);
	printf("%d\n", mat_1.length);
	block_size = 256;
	nblocks = (mat_1.length + block_size - 1)/block_size;
	cuda_array_arange<<<nblocks,block_size>>>(mat_1, 0);
	nblocks = (mat_2.length + block_size - 1)/block_size;
	cuda_array_arange<<<nblocks,block_size>>>(mat_2, 0);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
    		printf("Error: %s\n", cudaGetErrorString(err));
	/*for (int i = 0; i < mat_1.shape[0] ; i++)
	{
		for (int j = 0; j < mat_1.shape[1]; j++)
			printf("%4ld ", mat_1.nd_int64[get_index(mat_1, i, j)]);
		printf("\n");
	}*/
	//printf("\n");
	/*for (int i = 0; i < mat_2.shape[0] ; i++)
	{
		for (int j = 0; j < mat_2.shape[1]; j++)
			printf("%4ld ", mat_2.nd_int64[get_index(mat_2, i, j)]);
		printf("\n");
	}*/
	printf("\n");
	mat_r = cuda_array_create(2, a, nd_int64);
	nblocks = (mat_1.length + block_size - 1)/block_size;
	mat_prod<<<nblocks, block_size>>>(mat_r, mat_1, mat_2);
	cudaDeviceSynchronize();
	/*for (int i = 0; i < mat_r.shape[0] ; i++)
	{
		for (int j = 0; j < mat_r.shape[1]; j++)
			printf("%4ld ", mat_r.nd_int64[get_index(mat_r, i, j)]);
		printf("\n");
	}*/
	return (0);
}


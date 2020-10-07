#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ndarray.h"
#include <time.h>
#include <stdarg.h>

int *init_shape(int nd, ...)
{
    va_list ap;
    int *shape;

    va_start(ap, nd);
    shape = malloc(nd * sizeof(int));
    for (int i = 0; i < nd; i++)
    {
        shape[i] = va_arg(ap, int);
    }
    va_end(ap);
    return(shape);
}

int free_array(t_ndarray dump)
{
    free(dump.data.raw_data);
    free(dump.shape);
    free(dump.strides);
    return (1);
}

t_ndarray init_array(char *temp, int nd, int *shape, int type)
{
    t_ndarray a;

    a.type = type;
    a.nd = nd;
    a.shape = shape;
    a.strides = malloc(nd * sizeof(int));
    for (int i = 0; i < a.nd; i++)
    {
        a.strides[i] = 1;
        for (int j = i + 1; j < a.nd; j++)
            a.strides[i] *= a.shape[j];
        a.strides[i] *= type;
    }
    a.data.raw_data = calloc(a.shape[0] * a.shape[1] , a.type);
    if (temp)
        memcpy(a.data.raw_data, temp, a.shape[0] * a.shape[1] * a.type);
    return (a);
}

t_ndarray mat_product(t_ndarray mat1, t_ndarray mat2)
{
    t_ndarray mat_p;
    
    mat_p.type = sizeof(double); // this will be checked from the types of the matrices
    mat_p = init_array(NULL, 2, init_shape(2, mat1.shape[0], mat2.shape[1]), sizeof(double));

    
     for (int i = 0; i < mat1.shape[0]; ++i) {
      for (int j = 0; j < mat2.shape[1]; ++j) {
         for (int k = 0; k < mat1.shape[1]; ++k) {
            mat_p.data.double_nd[(j * mat_p.strides[1] + i * mat_p.strides[0])/mat_p.type]+= 
                        mat1.data.double_nd[(k * mat1.strides[1] + i * mat1.strides[0])/mat_p.type] * mat2.data.double_nd[(j * mat2.strides[1] + k * mat2.strides[0])/mat_p.type];
         }
      }
   }
   return (mat_p);
}

int main(void)
{
    int i;
    double m_1[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 7, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1};
    // int m_1_shape[] = {4,5};
    double m_2[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 7, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1};
    // int m_2_shape[] = {5,4};

    t_ndarray nd_arr_m1;
    t_ndarray nd_arr_m2;
    t_ndarray mat_p;

    /* init the fist matrix */
    nd_arr_m1 = init_array((char *)m_1, 2, init_shape(2, 4, 5), sizeof(double));

    /* init the second matrix */
    nd_arr_m2 = init_array((char *)m_2, 2, init_shape(2, 5, 4), sizeof(double));

    /* the product matrix time loop test*/
    // clock_t start, end;
    // double cpu_time_used;
    // int loops = 1000000;

    // start = clock();
    // for (i = 0; i < loops; i++)
    // {
    //     mat_p = mat_product(nd_arr_m1, nd_arr_m2);
    //     free_array(mat_p);
    // }
    // end = clock();
    // cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("\nlooped %d times in %fs\n", loops, cpu_time_used);
    
    /* the product matrix */
    mat_p = mat_product(nd_arr_m1, nd_arr_m2);
    printf("mat_p.shape : (%d,%d)\n", mat_p.shape[0], mat_p.shape[1]); 
    /* printing the result of the product */
    i = 0;
    while (i <  mat_p.shape[1]* mat_p.shape[0])
    {
        printf(" %f,", (mat_p.data.double_nd)[i]);
        i = i + 1;
        if (i % mat_p.shape[0] == 0) // skipping a line when acessing the next row
            printf("\n");
    }
    free_array(mat_p);
    free_array(nd_arr_m1);
    free_array(nd_arr_m2);
    return 0;
}
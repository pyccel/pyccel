#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ndarray_01.h"

t_ndarray init_array(double *temp, int nd, int *shape, int type)
{
    t_ndarray a;

    // int i;

    // i = -1;
    // while(++i < 7)
    // {
    //     printf("==a.data : %d\n", ((int*)temp)[i]);
    // }
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
    if (temp)
    {
        a.data = malloc(a.shape[0] * a.shape[1] * a.type);
        memcpy(a.data, temp, a.shape[0] * a.shape[1] * a.type);
    }
    return (a);
}

t_ndarray mat_product(t_ndarray mat1, t_ndarray mat2)
{
    t_ndarray mat_p;
    double *a;
    double *b;
    double *c;

    mat_p.shape = malloc(2 * sizeof(int));
    mat_p.shape[1] = mat2.shape[1];
    mat_p.shape[0] = mat1.shape[0];
    mat_p = init_array(NULL, 2, mat_p.shape,sizeof(double));
    mat_p.data = malloc(mat_p.shape[1] * mat_p.shape[0] * mat1.type); // allocation the raw data buffer
    memset(mat_p.data, 0, mat_p.shape[1] * mat_p.shape[0] * mat1.type);
    a = (double *)mat_p.data;
    b = (double *)mat1.data;
    c = (double *)mat2.data;
//     printf("a %f - \n", (double)(*mat1.data));
//      for (int i = 0; i < mat1.shape[0]; ++i) {
//       for (int j = 0; j < mat1.shape[1]; ++j) {
//         printf("%f - \n", b[j+ i*mat1.shape[1]]);
//       }
//    }
     for (int i = 0; i < mat1.shape[0]; ++i) {
      for (int j = 0; j < mat2.shape[1]; ++j) {
         for (int k = 0; k < mat1.shape[1]; ++k) {
                a[j + i * mat_p.shape[1]]+= 
                        b[k + i * mat1.shape[1]] * c[j + k * mat2.shape[1]];
         }
      }
   }

   return (mat_p);
}


int main(void)
{
    int i;

    t_ndarray nd_arr_m1;
    t_ndarray nd_arr_m2;
    t_ndarray mat_p;

    int tmp_shape_1[] = {5, 4};
    double m_1[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 7, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1};
    nd_arr_m1 = init_array(m_1, 2, tmp_shape_1, sizeof(double));
    double m_2[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 7, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1};
    int tmp_shape_2[] = {4, 5};
    nd_arr_m2 = init_array(m_2, 2, tmp_shape_2, sizeof(double));

    // printf("%f\n", (double)nd_arr_m1.data[0]);
    /* the product matrix */
    for (i = 0; i < 1000000; i++)
    {
        mat_p = mat_product(nd_arr_m1, nd_arr_m2);
        free(mat_p.data);
    }
    i = 0;
    printf("mat_p.shape : (%d,%d)\n", mat_p.shape[0], mat_p.shape[1]); // the shape is stored inversed
        mat_p = mat_product(nd_arr_m1, nd_arr_m2);
    
    /* printing the result of the product */
    double *n = (double *)mat_p.data;
    while (i <  mat_p.shape[1] * mat_p.shape[0])
    {
        printf(" %f,", n[i]);
        i = i + 1;
        if (i % mat_p.shape[1] == 0) // skipping a line when acessing the next row
            printf("\n");
    }
    // int s = mat_p.shape[1] * mat_p.shape[0];
    // for(i = 0; i < s; ++i)
    // {
    //     printf(" %f,", n[i]);
    // }
    return 0;
}
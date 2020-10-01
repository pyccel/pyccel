#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ndarray.h"

t_ndarray mat_product(t_ndarray mat1, t_ndarray mat2)
{
    t_ndarray mat_p;

    memset(mat_p.shape, -1, sizeof(mat_p.shape));
    mat_p.shape[0] = mat2.shape[0];
    mat_p.shape[1] = mat1.shape[1];
    mat_p.type = sizeof(double); // this will be checked from the types of the matrices
    mat_p.buffer.raw_data = malloc(mat_p.shape[0] * mat_p.shape[1] * mat_p.type); // allocation the raw data buffer

    
     for (int i = 0; i < mat1.shape[1]; ++i) {
      for (int j = 0; j < mat2.shape[0]; ++j) {
         for (int k = 0; k < mat1.shape[0]; ++k) {
            mat_p.buffer.double_nd[j + i * mat_p.shape[0]]+= 
                        mat1.buffer.double_nd[k + i * mat1.shape[0]] * mat2.buffer.double_nd[j + k * mat2.shape[0]];
         }
      }
   }

   return (mat_p);
}


int main(void)
{
    int i;
    double m_1[2][3] = {2, 3, 5, 5, 6, 7};
    double m_2[3][4] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 13, 14, 10};

    t_ndarray nd_arr_m1;
    t_ndarray nd_arr_m2;

    /* init the fist matrix */
    nd_arr_m1.buffer.raw_data = malloc(sizeof(m_1)); // allocation the raw data buffer
    memcpy(nd_arr_m1.buffer.raw_data, m_1, sizeof(m_1)); // copy the data to the ndarray
    /* TODO : make func that init the shape */
    memset(nd_arr_m1.shape, -1, sizeof(nd_arr_m1.shape));
    nd_arr_m1.shape[0] = 3;
    nd_arr_m1.shape[1] = 2;

    /* init the second matrix */
    nd_arr_m2.buffer.raw_data = malloc(sizeof(m_2)); // allocation the raw data buffer
    memcpy(nd_arr_m2.buffer.raw_data, m_2, sizeof(m_2));
    /* TODO : make func that init the shape */
    memset(nd_arr_m2.shape, -1, sizeof(nd_arr_m2.shape));
    nd_arr_m2.shape[0] = 4;
    nd_arr_m2.shape[1] = 3;

    /* the product matrix */
    t_ndarray mat_p = mat_product(nd_arr_m1, nd_arr_m2);
    i = 0;
    printf("mat_p.shape : (%d,%d)\n", mat_p.shape[1], mat_p.shape[0]); // the shape is stored inversed
    
    /* printing the result of the product */
    while (i <  mat_p.shape[1]* mat_p.shape[0])
    {
        printf(" %f,", mat_p.buffer.double_nd[i]);
        i = i + 1;
        if (i % mat_p.shape[0] == 0) // skipping a line when acessing the next row
            printf("\n");
    }
    free(mat_p.buffer.raw_data);
    free(nd_arr_m2.buffer.raw_data);
    free(nd_arr_m1.buffer.raw_data);

    return 0;
}
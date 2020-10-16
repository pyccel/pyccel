#include "ndarray.h"


t_ndarray *mat_product(t_ndarray *mat1, t_ndarray *mat2)
{
    t_ndarray *mat_p;
    int shape[2] = {mat1->shape[0], mat2->shape[1]};
    
    mat_p = init_array(NULL, 2, shape, sizeof(double));

    
     for (int i = 0; i < mat1->shape[0]; ++i) {
      for (int j = 0; j < mat2->shape[1]; ++j) {
         for (int k = 0; k < mat1->shape[1]; ++k) {
            mat_p->data->double_nd[get_index(mat_p, i, j)]+= 
                        mat1->data->double_nd[get_index(mat1, i, k)] * mat2->data->double_nd[get_index(mat2, k, j)];
         }
      }
   }
   return (mat_p);
}

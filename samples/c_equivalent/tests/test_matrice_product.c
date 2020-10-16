#include "../ndarray.h"

int main(void)
{
    double m_1[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 7, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1};
    int m_1_shape[] = {4,5};
    double m_2[] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 260, 6.34, 7, 8.002, 0.056, 45, 0.1, 1.02, 0.25, 0.00005, 1};
    int m_2_shape[] = {5,4};

    t_ndarray *m1;
    t_ndarray *m2;
    t_ndarray *mat_p;

    /* init the fist matrix */
    m1 = init_array((char *)m_1, 2, m_1_shape, sizeof(double));

    /* init the second matrix */
    m2 = init_array((char *)m_2, 2, m_2_shape, sizeof(double));

    /* the product matrix time loop test*/
    // clock_t start, end;
    // double cpu_time_used;
    // int loops = 1000000;

    // start = clock();
    // for (i = 0; i < loops; i++)
    // {
    //     mat_p = mat_product(m1, m2);
    //     free_array(mat_p);
    // }
    // end = clock();
    // cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("\nlooped %d times in %fs\n", loops, cpu_time_used);
    
    /* the product matrix */
    mat_p = mat_product(m1, m2);
    printf("mat_p.shape : (%d,%d)\n", mat_p->shape[0], mat_p->shape[1]); 
    /* printing the result of the product */
    array_data_dump(mat_p);
    free_array(mat_p);
    free_array(m1);
    free_array(m2);
    return 0;
}
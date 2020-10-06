#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ndarray_01.h"


t_ndarray init_array(void *temp, int nd, int *shape, int type)
{
    t_ndarray a;

    // int i;

    // i = -1;
    // while(++i < 7)
    // {
    //     printf("==a.data : %d\n", ((int*)temp)[i]);
    // }
    a.type = type;
    printf("type : %d\n", type);
    a.nd = nd;
    a.shape = shape;
    a.strides = malloc(a.nd * sizeof(int));
    for (int i = 0; i < a.nd; i++)
    {
        printf("%d\n", i);
        a.strides[i] = 1;
        for (int j = i + 1; j < a.nd; j++)
            a.strides[i] *= a.shape[j];
        a.strides[i] *= type;
    }
    a.data = temp;
    printf("%f\n", (double)a.data[0]);
    return (a);
}

int main(void)
{
    int i;
    int int_arr[] = {2, 3, 5, 5, 6, 7};
    int shape[] = {2, 3};

    t_ndarray a;

    a = init_array(int_arr, 2, shape, sizeof(int));
    i = 0;
    printf("=a.data : %d\n", (int)a.strides[0]);
    printf("=a.data : %d\n", (int)a.strides[1]);
    printf("=a.data : %d\n", (int)a.data[a.strides[0] * 1 + a.strides[1] * 2]);
    while (i  < a.shape[0] * a.shape[1] * a.type)
    {
        printf("a.data : %d - %d\n", (int)a.data[i], i);
        i = i + a.strides[1];
    }
    return 0;
}
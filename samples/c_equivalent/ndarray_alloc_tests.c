#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ndarray.h"

int main(void)
{
    int i;
    double double_arr[2][3] = {2, 3, 5, 5, 6, 7};
    int int_arr[3][2] = {2, 3, 5, 5, 6, 7};
    int int_arr_3d[3][2][2] = {2, 3, 5, 5, 6, 7, 10, 11, 12, 13, 14, 255};

    t_ndarray_type a;

    printf("testing float\n");
    a.raw_data = malloc(6 * sizeof(double));
    printf("size of arr %lu - %lu\n", sizeof(double_arr), sizeof(double_arr[0]));
    memcpy(a.raw_data, double_arr, sizeof(double_arr));

    printf("testing access with cords %f [0][2] === %f\n\n",a.double_nd[2 + 1 * 3], double_arr[1][2]); // index = y + x * width
    i = 0;
    while (i < 6)
    {
        printf("%f\n",a.double_nd[i]);
        i = i + 1;
    }
    free(a.raw_data);
    
    printf("\n\ntesting int\n");
    a.raw_data = malloc(6 * sizeof(int));
    printf("size of arr %lu\n", sizeof(int_arr));
    memcpy(a.raw_data, int_arr, sizeof(int_arr));
    printf("testing access with cords %d [2][0] === %d\n\n",a.int_nd[0 + 2 * 2], int_arr[2][0]);
    i = 0;
    while (i < 6)
    {
        printf("%d\n",a.int_nd[i]);
        i = i + 1;
    }
    free(a.raw_data);
    
    
    printf("\n\ntesting 3d int array\n");
    a.raw_data = malloc(6 * sizeof(int));
    printf("size of arr %lu\n", sizeof(int_arr_3d));
    memcpy(a.raw_data, int_arr_3d, sizeof(int_arr_3d));
    printf("testing access with cords %d [2][0] === %d\n\n",a.int_nd[1 + 2 * 1 + 2*2*2], int_arr_3d[2][1][1]); // index = y + x * width + z*width*height
    i = 0;
    while (i < 12)
    {
        printf("%d\n",a.int_nd[i]);
        i = i + 1;
    }
    free(a.raw_data);
    return 0;
}
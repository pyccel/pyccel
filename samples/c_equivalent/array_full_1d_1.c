#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(void)
{
    double *a;
    int i;
    
    a = malloc(sizeof(double) * 4);
    for (i = 0; i < 4; ++i)
        a[i] = 5;
    
    for (i = 0; i< 4; ++i)
    {
        printf("%f\n",a[i]);
    }
    free(a);
    return 0;
}
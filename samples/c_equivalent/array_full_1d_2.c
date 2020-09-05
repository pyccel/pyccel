#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(void)
{
    double *a;
    int i;

    double b[] = {5 , 5, 5, 5};
    a = malloc(sizeof(double) * 4);
    memcpy(a, b, sizeof(b));

    for (i = 0; i< 4; ++i)
    {
        printf("%f\n",a[i]);
    }
    free(a);
    return 0;
}
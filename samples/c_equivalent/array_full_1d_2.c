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

    i = 0;
    while (i < 4)
    {
        printf("%d\n",a[i]);
        i = i + 1;
    }
    free(a);
    return 0;
}
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(void)
{
    double *a;

    double holder[] = {2, 3, 4, 1};
    a = malloc(sizeof(holder));
    memcpy((void *)a, (void *)holder, sizeof(holder));

    for (int i = 0; i<sizeof(holder) / sizeof(double); ++i)
    {
        printf("%f\n",a[i]);
    }
    free(a);
    return 0;
}
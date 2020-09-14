#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(void)
{
    int *a;
    int i;

    a = malloc(sizeof(int) * 4);

    i = 0;
    while (i < 4)
    {
        printf("%d\n",a[i]);
        i = i + 1;
    }
    free(a);
    return 0;
}
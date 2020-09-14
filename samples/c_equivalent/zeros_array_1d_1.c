#include <stdlib.h>
#include <string.h>
#include <stdio.h>



int main(void)
{
    double *a;
    int i;
    
    a = malloc(sizeof(double) * 4);
    for (i = 0; i < 4; ++i)
        a[i] = 0;
    
    i = 0;
   while(i < 4)
    {
        printf("%g\n",a[i]);
        i = i + 1;
    }
    free(a);
    return 0;
}
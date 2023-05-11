#ifndef LISTS_H
#define LISTS_H

#include <stdlib.h>
#include <Python.h>
#include <stdio.h>
#include <strings.h>
#include <unistd.h>

typedef struct 
{
    size_t value;
    void *pointer;
    char *type;
}   GenericObject;

typedef struct
{
    GenericObject **items;
    size_t noi;
    size_t capacity;
}   PyccelList;

PyccelList *initialise_list(size_t noi);
void    initialise_genericobject(GenericObject **obj, char  *type, size_t value, void *pointer);

int     pyccel_append(PyccelList **list, size_t value, void *pointer);
int     pyccel_expandlist(PyccelList **list);
int     pyccel_copylist(PyccelList **src, PyccelList **dest);

void    print_list(PyccelList *list);
void    fill_itemswithvalues(PyccelList **list, size_t value);

void    free_list(PyccelList *list);
void    free_genericobject(GenericObject *go);

#endif
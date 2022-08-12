/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#ifndef LISTS_H
#define LISTS_H

#include <stdlib.h>
#include <stdbool.h>

#define DEFAULT_CAP 10

typedef enum e_type
{
    lst_bool     ,
    lst_int8     ,
    lst_int16    ,
    lst_int32    ,
    lst_int64    ,
    lst_float    ,
    lst_double   ,
    lst_cfloat   ,
    lst_cdouble  ,
    lst_list
};

typedef struct  s_elem
{
    void*       raw_data;
    enum e_type         type;
}               t_elem;

typedef struct  s_list
{
    t_elem**   elements;
    size_t  capacity;
    size_t  size;
}               t_list;

t_list   *allocate_list(size_t size);
t_list   *fill_list(t_list *list, ...);
void*    append(t_list* list_1, t_list* list_2);
void     clear(t_list* list);
t_list*  copy(t_list* list);
size_t   count(t_list* list);
void     extend(t_list* list, void* object);
size_t   index(t_list* list, void* value, int start, int stop);
void     insert(t_list* list, int index, void* object);
void*    pop(t_list* list, int index);
void     remove(t_list* list, void* value);
void     reverse(t_list* list);
void     sort(t_list* list, bool reverse);
#endif

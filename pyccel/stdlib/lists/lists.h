
#ifndef _LISTS_
#define _LISTS_
#define DEFAULT_CAP 10

#include <stddef.h>
#include <string.h>
#include <stdlib.h>

// typedef enum e_type t_type;

typedef enum    e_type
{
    lst_bool    ,
    lst_int8    ,
    lst_int16   ,
    lst_int32   ,
    lst_int64   ,
    lst_float   ,
    lst_double  ,
    lst_list
}       t_type;

typedef struct  s_list
{
    void*       elements;
    enum e_type type;
    size_t      capacity;
    size_t      size;
}               t_list;

typedef  int(*cmp_function)(void*, void*, t_type);

t_list  *allocate_list(size_t size, t_type type, void *elemnts);
void    free_list(t_list **list);
void    append(t_list** list1, t_list* list2);
void    clear(t_list* list);
t_list* copy(t_list* list);
size_t  count(t_list* list, void *item);
void    extend(t_list** list, void* item);
int     lst_index(t_list* list, void* item);
void    insert(t_list** list, size_t index, void* item);
void    pop(t_list* list, size_t index);
void    lst_remove(t_list* list, void* value);
void    reverse(t_list* list);
// void    sort(t_list* list, cmp_function);
int     default_cmp_func(void* item1, void* item2, t_type type);

void*   array_subscripting(t_list *list, size_t index);

#endif
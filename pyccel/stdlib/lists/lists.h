#define DEFAULT_CAP 10

#include <stddef.h>
#include <string.h>
#include <stdlib.h>

typedef enum e_type t_type;

typedef struct  s_list
{
    void*       elements;
    t_type      type;
    size_t      capacity;
    size_t      size;
}               t_list;

enum    e_type
{
    lst_bool   = 1  ,
    lst_int8   = 1  ,
    lst_int16  = 2  ,
    lst_int32  = 4  ,
    lst_int64  = 8  ,
    lst_float  = 4  ,
    lst_double = 8  ,
    lst_list   = sizeof(t_list)
};


typedef  int(*cmp_function)(void*, void*, t_type);


t_list   *allocate_list(size_t size, t_type type, void *elemnts);
int     free_list(t_list **list);
void    append(t_list** list1, t_list* list2);
void     clear(t_list* list);
t_list*  copy(t_list* list);
size_t   count(t_list* list, void *item);
void     extend(t_list* list, void* object); // redundant
int   lst_index(t_list* list, void* value);
void     insert(t_list** list, size_t index, void* object);
void     pop(t_list* list, size_t index);
void     lst_remove(t_list* list, void* value);
void     reverse(t_list* list);
void     sort(t_list* list, cmp_function);
int      default_cmp_func(void* item1, void* item2, t_type type); // hmm

void*   array_subscripting(t_list *list, size_t index);
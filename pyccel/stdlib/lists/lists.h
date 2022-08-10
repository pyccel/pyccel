#ifndef LISTS_H
# define LISTS_H

# include <stdlib.h>
# include <stdbool.h>
# include <Python.h>

typedef struct  s_node
{
    void    *ptr;
    // Need to know the type of the object pointed to by 'ptr'
}               t_node;

typedef struct  s_list
{
    t_node  **nodes;
    size_t  counter;  // Current nodes count in the buffer
    size_t  size;  // Total size of the buffer
}               t_list;


/** Utilities **/
t_list *list_alloc(size_t size);
void    list_free(t_list *list);


/** List functions **/
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
void     sort(t_list* list, /*key*/ bool reverse);

#endif

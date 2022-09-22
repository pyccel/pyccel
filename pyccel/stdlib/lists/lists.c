#include "lists.h"


t_list   *allocate_list(size_t size, t_type type, void *elemnts) // va_arg could alow us to take in multiple list of elements
{
    t_list *list;

    if (!(list = (t_list*)malloc(sizeof(t_list))))
        return NULL;
    list->size = size * type;
    list->capacity = DEFAULT_CAP;
    while (list->capacity <= list->size)
        list->capacity *= 2;
    if (list->size == 0) // maybe a useless check
        list->elements = NULL;
    else if (!(list->elements = malloc(list->capacity * type)))
        return NULL;
    memcpy(list->elements, elemnts, type * list->size);
    list->type = type;
    return list;
}

void     free_list(t_list **list)
{
    free((*list)->elements);
    free(*list);
    *list = NULL;
}

void    append(t_list** list1, t_list* list2)
{
    size_t totalSize = (*list1)->size + list2->size;
    char *elements = NULL;
    if (totalSize >= (*list1)->capacity)
    {
        t_list *tmp = *list1;
        elements = malloc(totalSize);
        memcpy(elements, (*list1)->elements, (*list1)->size);
        memcpy(&elements[(*list1)->size - 1], list2->elements, list2->size);
        free((*list1)->elements);
        *list1 = allocate_list(totalSize/(*list1)->type, (*list1)->type, elements);
        free(tmp);
        free(elements);
    }
    else
    {
        elements = (char*)(*list1)->elements;
        memcpy(&elements[(*list1)->size], list2->elements, list2->size);
    }
}

void     clear(t_list* list)
{
    list->size = 0;
}

t_list*  copy(t_list* list)
{
    return (allocate_list(list->size, list->type, list->elements));
}

size_t   count(t_list* list, void *item)
{
    size_t index = 0;
    int count = 0;
    char *elements = list->elements;
    printf("omok\n");
    while (index < list->size)
    {
        printf("and round we go\n");
        if (list->type == lst_list)
        {
            t_list *tmp = (t_list*)&(elements[index]);
            if (!(memcmp(item, tmp, sizeof(t_list))))
                count += 1;
        }
        else if (!(memcmp(item, &(elements[index]), list->type)))
                count += 1;
        index += list->type;
    }
    return (count);
}

void     extend(t_list* list, void* object){} // bassicaly append does that for us

int   lst_index(t_list* list, void* item)
{
    size_t index = 0;
    char *elements = (char*)(list->elements);
    while (index < list->size)
    {
        if (list->type == lst_list)
        {
            t_list *tmp = (t_list*)(&elements[index]);
            if (!(memcmp(item, tmp, sizeof(t_list))))
                return index/list->type;
        }
        else if (!(memcmp(item, &elements[index], list->type)))
                return index/list->type;
        index += list->type;
    }
    return (-1);
}

void     insert(t_list** list, size_t index, void* object)
{
    size_t totalSize = (*list)->size + 1;
    char *elements = NULL;
    if (totalSize >= (*list)->capacity)
    {
        t_list *tmp = *list;
        elements = malloc((*list)->type * totalSize);
        memcpy(elements, (*list)->elements, (*list)->size);
        memcpy(&elements[(*list)->size], object, 1);
        *list = allocate_list(totalSize, (*list)->type, elements);
        free(tmp);
    }
    else
    {
        elements = (char*)(*list)->elements;
        memcpy(&elements[(*list)->size], object, 1);
    }
}

void    pop(t_list* list, size_t index)
{
    if (list->size/list->type > index)
    {
        char *elements = (char*)(list->elements);
        while ((index + (2 * list->type)) < list->size)
        {
            memcpy(&elements[index], &elements[index + list->type], list->type);
            index += list->type;
        }
        list->size--;
    }
}

void     lst_remove(t_list* list, void* value)
{
    int index;
    if ((index = lst_index(list, value)) != -1)
        pop(list, (size_t)index);
}

void     reverse(t_list* list)
{
    char *elements = (char*)(list->elements);
    void *buffer = malloc(list->type);
    size_t index = 0;
    while (index + list->type <= list->size/2)
    {
        memcpy(buffer, &elements[index], list->type);
        memcpy(&elements[index], &elements[list->size - index - list->type], list->type);
        memcpy(&elements[list->size - index - list->type], buffer, list->type);
        index += list->type;
    }
    free(buffer);
}


void*   array_subscripting(t_list *list, size_t index)
{
    if (list->size/list->type > index)
    {
        char *elements = (char*)(list->elements);
        return ((void*)(&elements[index * list->type]));
    }
    return (NULL);
}
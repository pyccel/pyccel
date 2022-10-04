#include "lists.h"

int tSizes[8] = {
            1,
            sizeof(int8_t), 
            sizeof(int16_t), 
            sizeof(int32_t), 
            sizeof(int64_t), 
            sizeof(float), 
            sizeof(double), 
            sizeof(void *)
};

t_list   *allocate_list(size_t size, t_type type, void *elemnts) // va_arg could alow us to take in multiple list of elements
{
    t_list *list;

    if (!(list = (t_list*)malloc(sizeof(t_list))))
        return NULL;
    list->size = size * tSizes[type];
    list->type = type;
    list->capacity = DEFAULT_CAP;
    while (list->capacity <= list->size)
        list->capacity *= 2;
    if (list->size == 0)
        list->elements = NULL;
    else if (!(list->elements = malloc(list->capacity * tSizes[type])))
        return NULL;
    if (elemnts)
        memcpy(list->elements, elemnts, tSizes[type] * list->size);
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
        memcpy(&elements[(*list1)->size], list2->elements, list2->size);
        free((*list1)->elements);
        *list1 = allocate_list(totalSize/tSizes[(*list1)->type], (*list1)->type, elements);
        free(tmp);
        free(elements);
    }
    else
    {
        elements = (char*)(*list1)->elements;
        memcpy(&elements[(*list1)->size], list2->elements, list2->size);
        (*list1)->size += tSizes[(*list1)->type];
    }
}

void     clear(t_list* list)
{
    list->size = 0;
}

t_list*  copy(t_list* list)
{
    return (allocate_list(list->size / tSizes[list->type], list->type, list->elements));
}

size_t   count(t_list* list, void *item)
{
    size_t index = 0;
    int count = 0;
    char *elements = list->elements;
    while (index < list->size)
    {
        if (list->type == lst_list)
        {
            t_list *tmp = (t_list*)&(elements[index]);
            if (!(memcmp(item, tmp, sizeof(t_list))))
                count += 1;
        }
        else if (!(memcmp(item, &(elements[index]), tSizes[list->type])))
                count += 1;
        index += tSizes[list->type];
    }
    return (count);
}

void     extend(t_list** list, void* item)
{
    size_t totalSize = (*list)->size + tSizes[(*list)->type];
    char *elements = NULL;
    if (totalSize > (*list)->capacity)
    {
        elements = malloc(totalSize);
        memcpy(elements, (*list)->elements, (*list)->size);
        memcpy(&elements[(*list)->size], item, 1);
        *list = allocate_list(totalSize, (*list)->type, elements);
        free(elements);
    }
    else
    {
        elements = (char*)(*list)->elements;
        memcpy(&elements[(*list)->size], item, 1);
    }
}

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
                return index/tSizes[list->type];
        }
        else if (!(memcmp(item, &elements[index], tSizes[list->type])))
                return index/tSizes[list->type];
        index += tSizes[list->type];
    }
    return (-1);
}

void     insert(t_list** list, size_t index, void* item) // needs a redo
{
    size_t totalSize = (*list)->size + tSizes[(*list)->type];
    index *= tSizes[(*list)->type];
    if (index >= (*list)->size)
        index = (*list)->size;
    char *elements = NULL;
    if (totalSize > (*list)->capacity)
    {
        elements = malloc(totalSize);
        memcpy(elements, &(*list)->elements, index);
        memcpy(&elements[index], item, tSizes[(*list)->type]);
        memcpy(&elements[index + tSizes[(*list)->type]], (int8_t*)(*list)->elements + index * (*list)->type, (*list)->size - index);
        *list = allocate_list(totalSize / (*list)->type, (*list)->type, elements);
    }
    else
    {
        elements = (*list)->elements;
        size_t size = totalSize;
        while (size > index)
        {
            memcpy(&elements[size], &elements[size - tSizes[(*list)->type]], (*list)->type);
            size -= tSizes[(*list)->type];
        }
        memcpy(&elements[index], item, tSizes[(*list)->type]);
    }
}

void    pop(t_list* list, size_t index)
{
    index *= tSizes[list->type];
    if (list->size > index)
    {
        char *elements = (char*)(list->elements);

        while ((index + tSizes[list->type]) <= list->size)
        {
            memcpy(&elements[index], &elements[index + tSizes[list->type]], tSizes[list->type]);
            index += tSizes[list->type];
        }
        list->size -= tSizes[list->type];
    }
}

void     lst_remove(t_list* list, void* item)
{
    int index;
    if ((index = lst_index(list, item)) != -1)
        pop(list, (size_t)index);
}

void     reverse(t_list* list)
{
    char *elements = (char*)(list->elements);
    void *buffer = malloc(tSizes[list->type]);
    size_t index = 0;
    while (index + tSizes[list->type] <= list->size/2)
    {
        memcpy(buffer, &elements[index], tSizes[list->type]);
        memcpy(&elements[index], &elements[list->size - index - tSizes[list->type]], tSizes[list->type]);
        memcpy(&elements[list->size - index - tSizes[list->type]], buffer, tSizes[list->type]);
        index += tSizes[list->type];
    }
    free(buffer);
}


void*   array_subscripting(t_list *list, size_t index)
{
    if (list->size/tSizes[list->type] > index)
    {
        char *elements = (char*)(list->elements);
        return ((void*)(&elements[index * tSizes[list->type]]));
    }
    return (NULL);
}
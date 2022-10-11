#include "lists.h"
#include <stdio.h>

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
    if (!(list->elements = malloc(list->capacity)))
        return NULL;
    if (elemnts)
        memcpy(list->elements, elemnts, list->size);
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

// Sorting ////////////////////////////

void set_length_and_type(t_list *list, size_t *len, int8_t *type)
{
    t_list *tmp;

    tmp = list;
    *len = tmp->size / tSizes[tmp->type];
    while (tmp->type == lst_list)
    {
        tmp = ((t_list **)(tmp->elements))[0];
        *len *= tmp->size / tSizes[tmp->type];
    }
    *type = tmp->type;
}

size_t collect_data(t_list *list, int8_t *group, size_t offset)
{
    size_t length;

    length = list->size / tSizes[list->type];
    if (list->type == lst_list)
    {
        for (size_t i = 0; i < length; i++)
            offset += collect_data(((t_list **)list->elements)[i], group, offset);
    }
    else
    {
        memcpy(group + offset*tSizes[list->type], list->elements, list->size);
        offset += length;
    }
    return (offset);
}

int8_t *group_node_items(t_list *list, size_t len, int8_t type)
{
    int8_t *group;

    group = (int8_t *)malloc(len * tSizes[type]);
    collect_data(list, group, 0);
    return (group);
}

int compare(t_list *list, int i1, int i2)
{
    int8_t *group_1;
    int8_t *group_2;
    int cmp;
    size_t len;
    int8_t type;

    if (list->type != lst_list)
    {
        switch (list->type)
        {
            case lst_int8:
                return *((int8_t *)GET_INDEX(list, i1)) - *((int8_t *)GET_INDEX(list, i2));
            case lst_int16:
                return *((int16_t *)GET_INDEX(list, i1)) - *((int16_t *)GET_INDEX(list, i2));
            case lst_int32:
                return *((int32_t *)GET_INDEX(list, i1)) - *((int32_t *)GET_INDEX(list, i2));
            case lst_int64:
                return *((int64_t *)GET_INDEX(list, i1)) - *((int64_t *)GET_INDEX(list, i2));
            case lst_float:
                return *((float *)GET_INDEX(list, i1)) - *((float *)GET_INDEX(list, i2));
            case lst_double:
                return *((double *)GET_INDEX(list, i1)) - *((double *)GET_INDEX(list, i2));
        }
    }

    cmp = 0;
    len = 0;
    type = 0;
    set_length_and_type(((t_list **)list->elements)[0], &len, &type);
    group_1 = group_node_items(((t_list **)list->elements)[i1], len, type);
    group_2 = group_node_items(((t_list **)list->elements)[i2], len, type);

    for (size_t i = 0; i < len && cmp == 0; i++)
    {
        switch (type)
        {
            case lst_int8:
                cmp = ((int8_t *)group_1)[i] - ((int8_t *)group_2)[i];
                break ;
            case lst_int16:
                cmp = ((int16_t *)group_1)[i] - ((int16_t *)group_2)[i];
                break ;
            case lst_int32:
                cmp = ((int32_t *)group_1)[i] - ((int32_t *)group_2)[i];
                break ;
            case lst_int64:
                cmp = ((int64_t *)group_1)[i] - ((int64_t *)group_2)[i];
                break ;
            case lst_float:
                cmp = ((float *)group_1)[i] - ((float *)group_2)[i];
                break ;
            case lst_double:
                cmp = ((double *)group_1)[i] - ((double *)group_2)[i];
                break ;
        }
    }
    free(group_1);
    free(group_2);
    return cmp;
}

int partition(t_list *list, int p, int r)
{
    int i;
    void *elements;
    int8_t tmp[64];
    size_t size;

    elements = list->elements;
    size = tSizes[list->type];
    i = p - 1;
    for (int j = p; j < r; j++)
    {
        if (compare(list, j, r) <= 0)
        {
            i++;
            memcpy(tmp, elements + i*size, size);
            memcpy(elements + i*size, elements + j*size, size);
            memcpy(elements + j*size, tmp, size);
            memset(tmp, 0, sizeof(tmp));
        }
    }
    memcpy(tmp, elements + (i+1)*size, size);
    memcpy(elements + (i+1)*size, elements + r*size, size);
    memcpy(elements + r*size, tmp, size);
    memset(tmp, 0, sizeof(tmp));
    return i+1;
}

void quicksort(t_list *list, int p, int r)
{
    int q;

    if (p < r)
    {
        q = partition(list, p, r);
        quicksort(list, p, q - 1);
        quicksort(list, q + 1, r);
    }
}

void sort(t_list *list)
{
    int8_t tmp[64] = {0};
    size_t len;
    size_t size;

    size = tSizes[list->type];
    len = list->size / size;
    if (len == 2 && compare(list, 0, 1) > 0)
    {
        memcpy(tmp, list->elements, size);
        memcpy(list->elements, list->elements + size, size);
        memcpy(list->elements + size, tmp, size);
    }
    else if (len > 2)
        quicksort(list, 0, (list->size / tSizes[list->type]) - 1);
}

///////////////////////////////////////

void print_list(t_list *list, int newline)
{
    printf("[");
    if (list->type == lst_list)
    {
        for (int i = 0; i < list->size / tSizes[list->type]; i++)
        {
            print_list(((t_list **)list->elements)[i], 0);
            if (i+1 < list->size / tSizes[list->type])
                printf(", ");
        }
    }
    else
    {
        for (int i = 0; i < list->size / tSizes[list->type]; i++)
        {
            switch(list->type)
            {
                case lst_int8:
                case lst_int16:
                case lst_int32:
                case lst_int64:
                    printf("%ld", *(int64_t *)GET_INDEX(list, i));
                    break;
                case lst_float:
                    printf("%f", *(float *)GET_INDEX(list, i));
                    break;
                case lst_double:
                    printf("%lf", *(double *)GET_INDEX(list, i));
            }
            if (i+1 < list->size / tSizes[list->type])
                printf(", ");
        }
    }
    printf("]");
    if (newline)
        printf("\n");
}
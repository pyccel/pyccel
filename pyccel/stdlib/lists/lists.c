#include "lists.h"


t_list   *allocate_list(size_t size, t_type type, void *elemnts)
{
    t_list *list;
    size_t tsize = tSizes[type];

    if (!(list = (t_list*)malloc(sizeof(t_list))))
        return NULL;
    list->size = size;
    list->type = type;
    list->capacity = DEFAULT_CAP;
    while (list->capacity <= list->size)
        list->capacity *= 2;
    if (!(list->elements = malloc(list->capacity * tsize)))
        return NULL;
    if (elemnts)
        memcpy(list->elements, elemnts, tsize * list->size);
    return list;
}

void     free_list(t_list **list)
{
    free((*list)->elements);
    free(*list);
    *list = NULL;
}

void    extend(t_list* list1, t_list* list2)
{
    size_t totalSize = list1->size + list2->size;
    size_t tsize = tSizes[list1->type];
    char *elements = list1->elements;

    if (totalSize >= list1->capacity)
    {
        while (list1->capacity <= totalSize)
            list1->capacity *= 2;
        elements = realloc(elements, list1->capacity * tsize);
        list1->elements = elements;
    }
    memcpy(&elements[list1->size], list2->elements, list2->size * tsize);
    list1->size = totalSize;
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
    while (index < list->size)
    {
        if (list->type == lst_list)
        {
            t_list *tmp = ((t_list**)elements)[index];
            if (!(memcmp(item, tmp, sizeof(t_list))))
                count += 1;
        }
        else if (!(memcmp(item, &(elements[index * tSizes[list->type]]), tSizes[list->type])))
                count += 1;
        index += 1;
    }
    return (count);
}

void     append(t_list* list, void* item)
{
    t_list listTmp;
    listTmp.capacity = DEFAULT_CAP;
    listTmp.elements = item;
    listTmp.size = 1;
    listTmp.type = list->type;
    extend(list, &listTmp);
}



int   lst_index(t_list* list, void* item)
{
    size_t index = 0;
    char *elements = (char*)(list->elements);
    while (index < list->size)
    {
        if (list->type == lst_list)
        {
            t_list *tmp = ((t_list**)elements)[index];
            t_list *itmTmp = (t_list*)item;
            // NEEDS TO COMPARE LITERALS (group node items)
            if (tmp->size == itmTmp->size || 
                !(memcmp(itmTmp->elements, tmp->elements, tmp->size * tSizes[tmp->type]))) 
                return index;
        }
        else if (!(memcmp(item, &elements[index * tSizes[list->type]], tSizes[list->type])))
                return index;
        index += 1;
    }
    return (-1);
}

size_t calculate_index(long int index, size_t size)
{
    if (index >= size)
        index = size;
    else if (index < 0)
    {
        index += size + 1;
        index = (index > 0) ? index : 0;
    }
    return index;
}

void     insert(t_list* list, long int index, void* item)
{
    char * elements = list->elements;
    size_t totalSize = list->size + 1;
    size_t tsize = tsize;
    size_t ind = calculate_index(index, list->size) * tsize;

    if (totalSize >= list->capacity)
    {
        list->capacity *= 2;
        elements = realloc(elements, list->capacity * tsize);
        list->elements = elements;
    }
    memmove(&(elements[ind + tsize]), &(elements[ind]), (list->size * tsize) - ind);
    memcpy(&(elements[ind]), item, tsize);
    list->size = totalSize; 
}

void    *pop(t_list* list, long int index)
{

    size_t tsize = tSizes[list->type];
    char *elements = (char*)(list->elements);
    t_pop_ret *ret_val = malloc(sizeof(t_pop_ret));

    index = (index < 0) ? list->size + index : index;
    if (index >= 0 && list->size > index)
    {
        ret_val->type = list->type;
        if (list->type == lst_list)
            ret_val->raw = ((t_list**)elements)[index];
        else
        {
            ret_val->raw = malloc(list->type);
            memcpy(ret_val->raw, elements[index * tsize], tsize);
        }
        memmove(&elements[index * tsize], &elements[(index + 1) * tsize], list->size - index);
        list->size -= 1;
        return (ret_val);
    }

    return (NULL);
}

void    free_pop(t_pop_ret** pop_val)
{
    if (pop_val && (*pop_val))
    {
        free((*pop_val)->raw);
        free(*pop_val);
        *pop_val = NULL;
    }
}

void     lst_remove(t_list* list, void* item)
{
    int index;
    t_pop_ret *ret;

    if ((index = lst_index(list, item)) != -1)
        ret = pop(list, (size_t)index);
    free_pop(ret);
}

void     reverse(t_list* list)
{
    size_t tsize = tSizes[list->type];
    size_t index = 0;
    char *elements = (char*)(list->elements);
    void *buffer = malloc(tsize);

    while (index + 1 <= list->size / 2)
    {
        memcpy(buffer, &elements[index * tsize], tsize);
        memcpy(&elements[index * tsize], &elements[(list->size - index - 1) * tsize], tsize);
        memcpy(&elements[(list->size - index - 1) * tsize], buffer, tsize);
        index++;
    }
    free(buffer);
}


void*   array_subscripting(t_list *list, size_t index)
{
    char *elements = (char*)(list->elements);

    if (list->size > index)
        return ((void*)(&elements[index * tSizes[list->type]]));
    return (NULL);
}
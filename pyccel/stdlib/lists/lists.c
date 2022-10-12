#include "lists.h"

t_list   *allocate_list(size_t size, t_type type, void *elemnts) // va_arg could alow us to take in multiple list of elements
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

void     append(t_list* list, void* item) // this could just be an insert call
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

t_pop_ret   *pop(t_list* list, long int index)
{

    size_t tsize = tSizes[list->type];
    char *elements = (char*)(list->elements);
    t_pop_ret *ret_val = malloc(sizeof(t_pop_ret));

    index = (index < 0) ? list->size + index : index;
    if (index >= 0 && list->size > index)
    {
        ret_val->type = list->type;
        if (list->type == lst_list)
            ret_val->raw = (char *)((t_list**)elements)[index];
        else
        {
            ret_val->raw = malloc(list->type);
            memcpy(ret_val->raw, &elements[index * tsize], tsize);
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
    {
        ret = pop(list, (size_t)index);
        free_pop(&ret);
    }
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

// Sorting ////////////////////////////

void set_length_and_type(t_list *list, size_t *len, int8_t *type)
{
    t_list *tmp;

    tmp = list;
    *len = tmp->size;
    while (tmp->type == lst_list)
    {
        tmp = ((t_list **)(tmp->elements))[0];
        *len *= tmp->size;
    }
    *type = tmp->type;
}

size_t collect_data(t_list *list, int8_t *group, size_t offset)
{
    if (list->type == lst_list)
    {
        for (size_t i = 0; i < list->size; i++)
            offset += collect_data(((t_list **)list->elements)[i], group, offset);
    }
    else
    {
        memcpy(group + offset*tSizes[list->type], list->elements, list->size);
        offset += list->size;
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
    int8_t type;
    size_t len;
    int cmp;

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
            case lst_complex:
                fprintf(stderr, "Ordering of complex numbers is not supported.");
                exit(-1);
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
            case lst_complex:
                fprintf(stderr, "Ordering of complex numbers is not supported.");
                exit(-1);
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

void sort(t_list *list, size_t rev)
{
    int8_t tmp[64] = {0};
    size_t tsize;
    size_t len;

    tsize = tSizes[list->type];
    len = list->size;
    if (len == 2 && compare(list, 0, 1) > 0)
    {
        memcpy(tmp, list->elements, tsize);
        memcpy(list->elements, list->elements + tsize, tsize);
        memcpy(list->elements + tsize, tmp, tsize);
    }
    else if (len > 2)
        quicksort(list, 0, list->size - 1);

    if (rev)
        reverse(list);
}

///////////////////////////////////////

void print_list(t_list *list, int newline)
{
    printf("[");
    if (list->type == lst_list)
    {
        for (int i = 0; i < list->size; i++)
        {
            print_list(((t_list **)list->elements)[i], 0);
            if (i+1 < list->size)
                printf(", ");
        }
    }
    else
    {
        for (int i = 0; i < list->size; i++)
        {
            switch(list->type)
            {
                case lst_int8:
                    printf("%hhd", *(int8_t *)GET_INDEX(list, i));
                    break;
                case lst_int16:
                    printf("%hd", *(int16_t *)GET_INDEX(list, i));
                    break;
                case lst_int32:
                    printf("%d", *(int32_t *)GET_INDEX(list, i));
                    break;
                case lst_int64:
                    printf("%ld", *(int64_t *)GET_INDEX(list, i));
                    break;
                case lst_float:
                    printf("%f", *(float *)GET_INDEX(list, i));
                    break;
                case lst_double:
                    printf("%lf", *(double *)GET_INDEX(list, i));
                    break;
                case lst_complex:
                    double real = creal(*(complex *)GET_INDEX(list, i));
                    double imag = cimag(*(complex *)GET_INDEX(list, i));
                    printf("%lf%s%lfj", real, imag >= 0 ? "+" : "", imag);
                    break;
            }
            if (i+1 < list->size)
                printf(", ");
        }
    }
    printf("]");
    if (newline)
        printf("\n");
}
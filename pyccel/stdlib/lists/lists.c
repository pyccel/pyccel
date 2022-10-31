#include "lists.h"

t_list   *allocate_list(size_t size, t_type type, void *elemnts)
/*
        Allocates a new t_list object

        Parameters
        ----------
        size : size_t
            the number of items in the list

        type : t_type
            The type of the new list's elements

        elemnts : void*
            A pointer on a table containing the list's elements

        Returns
        =======
        list : t_list
*/
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
/*
        Deallocate a t_list object

        Parameters
        ----------
        list : t_list**
            a double pointer on the list to be freed
*/
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
    memcpy(&elements[list1->size * tsize], list2->elements, list2->size * tsize);
    list1->size = totalSize;
}

void     clear(t_list* list)
/*
        Clears a t_list without deallocation

        Parameters
        ----------
        list : t_list**
            a pointer on the list to be freed
*/
{
    list->size = 0;
}

t_list*  copy(t_list* list)
/*
        Creates a copy t_list object from another

        Parameters
        ----------
        list : t_list*
            a pointer on the list to be copied
        
        Returns
        =======
        list : t_list
*/
{
    return (allocate_list(list->size, list->type, list->elements));
}

size_t   count(t_list* list, void *item)
/*
        Counts the number of occurrences of a specific element

        Parameters
        ----------
        list : t_list*
            the t_list where to look for occurrences

        item : void*
            The element to look for

        Returns
        =======
        count : size_t
*/
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
/*
        Adds an element to the end of a t_list object

        Parameters
        ----------
        list : t_list*
            the t_list where to add the element

        item : void*
            the element to be added
*/
{
    size_t tsize = tSizes[list->type];
    char* elements = list->elements;

    if ((list->size + 1) >= list->capacity)
    {
        list->capacity *= 2;
        elements = realloc(list->elements, list->capacity * tsize);
        list->elements = elements;
    }
    if (list->type == lst_list)
        memcpy(&elements[list->size * tsize], &item, tsize);
    else
        memcpy(&elements[list->size * tsize], item, tsize);
    list->size += 1;
}



int   lst_index(t_list* list, void* item)
/*
        looks for the index of a specific element and returns it if it exists

        Parameters
        ----------
        list : t_list*
            the t_list where to search for the element

        item : void*
            the element to be indexed

        Returns
        =======
        index : int
*/
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
/*
        calculates negative indexes

        Parameters
        ----------
        index : long int
            the index to be resolved

        size : size_t
            the element to be indexed

        Returns
        =======
        index : size_t
*/
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
/*
        Insert an element in a specific index

        Parameters
        ----------
        list : t_list*
            t_list where to insert the element

        index : long int
            the index where to insert the element

        item : void*
            the element to be inserted
*/
{
    char * elements = list->elements;
    size_t totalSize = list->size + 1;
    size_t tsize = tSizes[list->type];
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
/*
        removes an element from a specific index and return it

        Parameters
        ----------
        list : t_list*
            the t_list from where to remove the element

        index : long int
            index of the element to be removed

        Returns
        =======
        ret_val : t_pop_ret
*/
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
/*
        Deallocate the t_pop_ret object resulting from a pop operation

        Parameters
        ----------
        pop_val : t_pop_ret**
            double pointer on the t_pop_ret object to be deallocated
*/
{
    if (pop_val && (*pop_val))
    {
        free((*pop_val)->raw);
        free(*pop_val);
        *pop_val = NULL;
    }
}

void     lst_remove(t_list* list, void* item)
/*
        removes an element by value

        Parameters
        ----------
        list : t_list*
            the t_list from where to remove the element

        index : long int
            the element to be removed
*/
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
/*
        reverse the order of elements in a t_list

        Parameters
        ----------
        list : t_list*
            the t_list to be reversed
*/
{
    size_t tsize = tSizes[list->type];
    size_t index = 0;
    char *elements = (char*)(list->elements);
    char buffer[tsize];

    memset(buffer, 0, tsize);
    while (index + 1 <= list->size / 2)
    {
        memcpy(buffer, &elements[index * tsize], tsize);
        memcpy(&elements[index * tsize], &elements[(list->size - index - 1) * tsize], tsize);
        memcpy(&elements[(list->size - index - 1) * tsize], buffer, tsize);
        index++;
    }
}

t_list *lst_slice(t_list *list, size_t start, size_t end, int step, int order)
/*
        Returns a new t_list containing a slice of another t_list

        Parameters
        ----------
        list : t_list*
            t_list from where to get the slice 

        start : size_t
            the starting index for the slice

        end : size_t
            the ending index for the slice (excluded)

        step : int
            the pace at which to take the element

        order : int
            the order of the slice
                1 : for reverse order
                0 : for original t_list order
        
        Returns
        =======
        slice : t_list
*/
{
    if (!step || start == end)
        return (NULL);

    size_t tsize = tSizes[list->type];
    size_t size = end - start;
    char *buff = calloc((size / step), tsize);
    char *elements = list->elements;
    size_t index = 0;

    while (index < end)
    {
        if (order)
            memcpy(&buff[(size - index - 1) * tsize], &elements[((index * step) + start) * tsize], tsize);
        else
            memcpy(&buff[index * tsize], &elements[(index + start) * step * tsize], tsize);
        index++;
    }
    t_list *slice = allocate_list(size, list->type, buff);
    free(buff);
    return (slice);
}

void*   array_subscripting(t_list *list, size_t index)
/*
        provides access to a specific element in a t_list

        Parameters
        ----------
        list : t_list*
            the t_list of the element to be accessed

        index : long int
            the element to be accessed
*/
{
    char *elements = (char*)(list->elements);

    if (list->size > index)
        return ((void*)(&elements[index * tSizes[list->type]]));
    return (NULL);
}

// Sorting ////////////////////////////

/**
 * @brief Set the length and type object for group allocation.
 * 
 * @param list 
 * @param len 
 * @param type 
 */
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

/**
 * @brief Branch recursively through every element in the list and copy 
 * it's value in the array pointed to by `int8_t *group`.
 * Returns the number of elements copied on every recursion.
 * 
 * @param list 
 * @param group 
 * @param offset 
 * @return size_t 
 */
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

/**
 * @brief Allocate and collect the sub-elements of a list.
 * 
 * @param list 
 * @param len 
 * @param type 
 * @return int8_t* 
 */
int8_t *group_node_items(t_list *list, size_t len, int8_t type)
{
    int8_t *group;

    group = (int8_t *)malloc(len * tSizes[type]);
    collect_data(list, group, 0);
    return (group);
}

/**
 * @brief If the list consists of pointers over other lists, it attempts
 * to collect the sub-elements of the lists pointed to by the indexs `i1`
 * and `i2`, putting them in two seperate allocated arrays, then compare 
 * every value.
 * Returns comparision value.
 * 
 * @param list 
 * @param i1 
 * @param i2 
 * @return int 
 */
int compare(t_list *list, int i1, int i2)
{
    int8_t *group_1;
    int8_t *group_2;
    int8_t type;
    size_t len;
    int cmp;

    cmp = 0;
    len = 0;
    type = 0;
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

/**
 * @brief Always take the last element of the list and use it as an anchor
 * for sorting the elements with values less than it's value, while updating 
 * the index where the anchor should exist, then swap the value in the index 
 * `r` (the current position of the anchor) with new found position, this 
 * ensures that the anchor is always in the right position.
 * Returns the index of the anchor.
 * 
 * @param list 
 * @param p 
 * @param r 
 * @return int 
 */
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

/**
 * @brief Implementation of quicksort algorithm.
 * step 1: find the index of the pivot point `q` using `partition`
 * step 2: run quicksort recursivly on the two sides
 *         from list[p .. q-1] and list[q+1 .. r]
 * 
 * @param list 
 * @param p 
 * @param r 
 */
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

/**
 * @brief Sort the list pointed to by `t_list *list` in ascending order,
 * or in descending order depending on the value of `size_t rev`.
 * It runs the quicksort algorithm on lists with size greater than 2, or simply 
 * swap the elements if they happen to be not sorted without going through 
 * the recursion of quicksort function.
 * 
 * @param list 
 * @param rev 
 */
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

    if (rev && len > 1)
        reverse(list);
}

///////////////////////////////////////

/**
 * @brief Prints recursively the list.
 * 
 * @param list 
 * @param newline 
 */
void print_list(t_list *list, int newline)
{
    printf("[");
    for (int i = 0; i < list->size; i++)
    {
        if (list->type == lst_list)
            print_list(((t_list **)list->elements)[i], 0);
        else
        {
            switch (list->type)
            {
                case lst_int8:   printf("%hhd", *(int8_t *)GET_INDEX(list, i));  break;
                case lst_int16:  printf("%hd",  *(int16_t *)GET_INDEX(list, i)); break;
                case lst_int32:  printf("%d",   *(int32_t *)GET_INDEX(list, i)); break;
                case lst_int64:  printf("%ld",  *(int64_t *)GET_INDEX(list, i)); break;
                case lst_float:  printf("%f",   *(float *)GET_INDEX(list, i));   break;
                case lst_double: printf("%lf",  *(double *)GET_INDEX(list, i));  break;
                case lst_complex: 
                    // double real = creal(*(complex *)GET_INDEX(list, i));
                    // double imag = cimag(*(complex *)GET_INDEX(list, i));
                    // printf("%lf%s%lfj", creal, imag >= 0 ? "+" : "", imag);
                    break;
            }
        }
        if (i+1 < list->size)
            printf(", ");
    }
    printf("]");
    if (newline)
        printf("\n");
}

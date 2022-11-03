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
 * @brief compare_literals takes pointers over literals and compares their values
 * depending on type.
 * 
 * @param arg_1 
 * @param arg_2 
 * @param type 
 * @return int 
 */
int compare_literals(void *arg_1, void *arg_2, void *type)
{
    int _type = *(int *)type;

    if (_type == lst_int8)
    {
        if (*((int8_t *)arg_1) < *((int8_t *)arg_2))
            return -1;
        if (*((int8_t *)arg_1) > *((int8_t *)arg_2))
            return 1;
    }
    else if (_type == lst_int16)
    {
        if (*((int16_t *)arg_1) < *((int16_t *)arg_2))
            return -1;
        if (*((int16_t *)arg_1) > *((int16_t *)arg_2))
            return 1;
    }
    else if (_type == lst_int32)
    {
        if (*((int32_t *)arg_1) < *((int32_t *)arg_2))
            return -1;
        if (*((int32_t *)arg_1) > *((int32_t *)arg_2))
            return 1;
    }
    else if (_type == lst_int64)
    {
        if (*((int64_t *)arg_1) < *((int64_t *)arg_2))
            return -1;
        if (*((int64_t *)arg_1) > *((int64_t *)arg_2))
            return 1;
    }
    else if (_type == lst_float)
    {
        if (*((float *)arg_1) < *((float *)arg_2))
            return -1;
        if (*((float *)arg_1) > *((float *)arg_2))
            return 1;
    }
    else if (_type == lst_double)
    {
        if (*((double *)arg_1) < *((double *)arg_2))
            return -1;
        if (*((double *)arg_1) > *((double *)arg_2))
            return 1;
    }
    return 0;
}

/**
 * @brief if the type of arg_1 and arg_2 is lst_list, this function branchs recursively
 * throught their elements intil it reache the innermost dimention to compare the 
 * literal values. Otherwise, consider arg_1 and arg_2 as literals and compare their
 * values. If the lists have the same values, than it compares occurding to the size
 * of the lists.
 * Returns comparision value.
 * 
 * @param arg_1
 * @param arg_2
 * @param type
 * @return int
 */
int compare(void *arg_1, void *arg_2, void *type)
{
    int     min;
    int     cmp;
    t_list  *lst_1;
    t_list  *lst_2;

    cmp = 0;
    if (*(int *)type != lst_list)
        cmp = compare_literals(arg_1, arg_2, type);
    else
    {
        lst_1 = *(t_list **)arg_1;
        lst_2 = *(t_list **)arg_2;
        min   = MIN(lst_1->size, lst_2->size);

        for (int i = 0; i < min && cmp == 0; i++)
            cmp = compare(GET_ELM(lst_1, i), GET_ELM(lst_2, i), &lst_1->type);

        if (lst_1->size != lst_2->size && cmp == 0)
            cmp = lst_1->size > lst_2->size ? 1 : -1;
    }
    return cmp;
}

/**
 * @brief sort the list pointed to by `t_list *list` in ascending order,
 * or in descending order depending on the value of `size_t rev`.
 * It uses the function qsort_r(3)
 * 
 * @param list 
 * @param rev 
 */
void sort(t_list *list, size_t rev)
{
    #if (defined __linux__ || defined __GNU__)
    qsort_r(list->elements, list->size, tSizes[list->type], compare, &list->type);
    #elif (defined _WIN32 || defined _WIN64 || defined __WINDOWS__)
    qsort_s(list->elements, list->size, tSizes[list->type], compare, &list->type);
    #endif

    if (rev)
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

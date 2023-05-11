#include "lists.h"

PyccelList *initialise_list(size_t noi)
{
    PyccelList *list;

    list = (PyccelList *)malloc(sizeof(PyccelList));
    list->capacity = noi*2;
    list->items = (GenericObject **)malloc(sizeof(GenericObject*)*list->capacity);
    
    for (size_t i = 0; i < list->capacity; i++)
    {
        list->items[i] = (GenericObject *)malloc(sizeof(GenericObject));
        initialise_genericobject(&(list->items[i]), "v", 0, NULL);
    }

    list->noi = noi;
    return list;
}

void    initialise_genericobject(GenericObject **obj, char  *type, size_t value, void *pointer)
{
    (*obj)->type = type;
    (*obj)->pointer = pointer;
    (*obj)->value = value;
}

void print_list(PyccelList *list)
{
    size_t i = 0;
    write(1, "[", 1);
    while (i < list->noi)
    {
        if (!strcmp(list->items[i]->type, "pl"))
        {
            print_list(list->items[i]->pointer);
        }
        else
        {
            printf("%zu", list->items[i]->value);
            fflush(stdout);
        }
        i++;
        if (i < list->noi)
            write(1, ",", 1);
    }
    write(1, "]", 1);
}

void fill_itemswithvalues(PyccelList **list, size_t value)
{
    size_t i = 0;

    while (i < (*list)->noi)
    {
        (*list)->items[i]->value = value;
        (*list)->items[i]->type = "v";
        i++;
    }
}

void    free_list(PyccelList *list)
{
    for (size_t i = 0; i < list->capacity; i++)
    {
        if (!strcmp((list)->items[i]->type, "pl"))
        {
            free_list(list->items[i]->pointer);
            free_genericobject(list->items[i]);
        }
        else
        {
            free_genericobject(list->items[i]);
        }
    }
    free(list->items);
    free(list);
}

void    free_genericobject(GenericObject *go)
{
    free(go);
}

int pyccel_copylist(PyccelList **src, PyccelList **dest)
{
    for (size_t i = 0; i < (*src)->noi; i++)
    {
        if (!strcmp((*src)->items[i]->type, "pl"))
        {
            PyccelList *item = (PyccelList *)((*src)->items[i]->pointer);
            (*dest)->items[i]->pointer = initialise_list(item->noi);
            (*dest)->items[i]->type = "pl";
            pyccel_copylist(&item, (PyccelList **)(&((*dest)->items[i]->pointer)));
        }
        else
        {
            initialise_genericobject(&((*dest)->items[i]), (*src)->items[i]->type, (*src)->items[i]->value, (*src)->items[i]->pointer);
        }
    }
    return 0;
}

int pyccel_expandlist(PyccelList **list)
{
    PyccelList *newlist;

    if ((*list) == NULL)
        return 1;

    newlist = initialise_list((*list)->capacity);
    pyccel_copylist(list, &newlist);
    free_list(*list);
    *list = newlist;

    return 0;
}

int pyccel_append(PyccelList **list, size_t value, void *pointer)
{
    if (list == NULL)
        return 1;
    GenericObject *item_to_fill;

    if ((*list)->capacity == (*list)->noi)
        pyccel_expandlist(list);

    item_to_fill = (*list)->items[(*list)->noi];
    if (pointer != NULL)
        initialise_genericobject(&item_to_fill, "pl", 0, pointer);
    else
        initialise_genericobject(&item_to_fill, "v", value, NULL);

    (*list)->noi++;
    return 0;
}
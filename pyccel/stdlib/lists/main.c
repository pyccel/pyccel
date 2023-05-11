#include "lists.h"

int main()
{
    PyccelList *my_list = initialise_list(3);
    PyccelList *copy;
    PyccelList *array_list;
    PyccelList *mixed_list;

    //manually creating a small list
    initialise_genericobject(&(my_list->items[0]), "v", 1, NULL);
    initialise_genericobject(&(my_list->items[1]), "pl", 0, initialise_list(3));
    initialise_genericobject(&(my_list->items[2]), "pl", 0, initialise_list(2));


    //list with only values
    array_list = my_list->items[1]->pointer;
    fill_itemswithvalues(&array_list, 2);

    //mixed value,array_list
    mixed_list = my_list->items[2]->pointer;
    initialise_genericobject(&(mixed_list->items[0]), "v", 3, NULL);
    initialise_genericobject(&(mixed_list->items[1]), "pl", 0, initialise_list(4));
    fill_itemswithvalues((PyccelList **)(&(mixed_list->items[1]->pointer)), 3);
    
    //simple stress test
    for (int i = 0; i<100; i++)
    {
        copy = initialise_list(i);
        fill_itemswithvalues(&copy, i);
        pyccel_append(&my_list, 0, copy);
        print_list(my_list);
        printf("\n");
    }
    free_list(my_list);

    return 0;
}
#include "Set_extensions.h"

#define Set_pop_macro(type, key)\
key Set_pop(type *l){\
    type##_iter itr = type##_begin(*&l);\
    type##_erase_at(*&l, itr);\
    return(*itr.ref);\
}

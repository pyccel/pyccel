#include <stc/priv/template.h>
#include <stdarg.h>

// This function represents a call to the .pop() method.
// Self: Class type (e.g., hset_int64_t).
// i_key: Data type of the elements in the set (e.g., int64_t).

static inline i_key _c_MEMB(_pop)(Self* self) {
    _c_MEMB(_iter) itr = _c_MEMB(_begin)(self); // Get iterator of the first element in the set using (_begin).
    if (itr.ref) 
    {
        i_key value = *(itr.ref);
        _c_MEMB(_erase_at)(self, itr); // Remove the element by value using "_erase_at".
        return value;
    }
    return *(itr.ref); // Return the element that is being popped.
}

/*
 * This function represents a call to the .union() method.
 * @param self : The set instance
 * @param n : The number of variadic arguments passed to the method.
 * @param ... : The variadic arguments. These are the other sets in which elements may be found
 */
static inline Self _c_MEMB(_union)(Self* self, int n, ...) {
    Self union_result = _c_MEMB(_clone)(*self);

    va_list args;
    va_start(args, n);

    for (int i=0; i<n; ++i) {
        Self* other = va_arg(args, Self*);
        c_foreach (elem, Self, *other)
            _c_MEMB(_insert)(&union_result, (*elem.ref));
    }

    va_end(args);

    return union_result;
}

/**
 * This function represents a call to the .intersection_update() method.
 * @param self : The set instance to modify.
 * @param other : The other set in which elements must be found.
 */
static inline void _c_MEMB(_intersection_update)(Self* self, Self* other) {
    _c_MEMB(_iter) itr = _c_MEMB(_begin)(self);
    while (itr.ref)
    {
        i_key val = (*itr.ref);
        if (_c_MEMB(_contains)(other, val)) {
            _c_MEMB(_next)(&itr);
        } else {
            itr = _c_MEMB(_erase_at)(self, itr);
        }
    }
}

#include <stc/priv/template2.h>

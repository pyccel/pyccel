#ifndef SET_EXTENSIONS_H
#define SET_EXTENSIONS_H


#define _c_MEMB(name) c_JOIN(i_type, name)

// This function represents a call to the .pop() method.
// i_type: Class type (e.g., hset_int64_t).
// i_key: Data type of the elements in the set (e.g., int64_t).

static inline i_key _c_MEMB(_pop)(i_type* self) {
    _c_MEMB(_iter) itr = _c_MEMB(_begin)(self); // Get iterator of the first element in the set using (_begin).
    if (itr.ref) 
    {
        i_key value = *(itr.ref);
        _c_MEMB(_erase_at)(self, itr); // Remove the element by value using "_erase_at".
        return value;
    }
    return *(itr.ref); // Return the element that is being popped.
}

/**
 * This function represents a call to the .intersection() method.
 * @param self : The set instance.
 * @param n : The number of variadic arguments passed to the method.
 * @param ... : The variadic arguments. These are the other sets in which elements must be found.
 */
static inline i_type _c_MEMB(_intersection)(i_type* self, int n, ...) {
    i_type intersection = _c_MEMB(_init)();

    i_type* others[n];

    va_list args;
    va_start(args, n);
    for (int i=0; i<n; ++i) {
        others[i] = va_arg(args, i_type*);
    }
    va_end(args);

    c_foreach (elem, i_type, self) {
        i_key val = (*elem.ref);
        bool in_intersection(true);
        int i=0;
        while (i<n && in_intersection) {
            in_intersection = in_intersection && _c_MEMB(_contains)(others[i], val);
            i++;
        }
        if (in_intersection) {
            _c_MEMB(_insert)(&intersection, val);
        }
    }

    return intersection;
}

#undef i_type
#undef i_key
#endif

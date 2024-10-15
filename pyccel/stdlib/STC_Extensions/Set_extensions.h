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
 * This function represents a call to the .intersection_update() method.
 * @param self : The set instance to modify.
 * @param other : The other set in which elements must be found.
 */
static inline void _c_MEMB(_intersection_update)(i_type* self, i_type* other) {
    _c_MEMB(_iter) itr = _c_MEMB(_begin)(self);
    while (itr != _c_MEMB(_end)(self))
    {
        i_key val = (*itr.ref);
        if (_c_MEMB(_contains)(others[i], val)) {
            _c_MEMB(_next)(&it)
        } else {
            itr = _c_MEMB(_erase_at)(&intersection, val);
        }
    }
}

#undef i_type
#undef i_key
#endif

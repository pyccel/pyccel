#ifndef LIST_EXTENSIONS_H
#define LIST_EXTENSIONS_H


#define _c_MEMB(name) c_JOIN(i_type, name)

// This function represents a call to the .pop() method.
// i_type: Class type (e.g., hset_int64_t).
// i_key: Data type of the elements in the set (e.g., int64_t).

static inline i_key _c_MEMB(_pull_elem)(i_type* self, intptr_t pop_idx) {
    // Get the iterator for the specified element using (_advance) and (_begin)
    _c_MEMB(_iter) itr = _c_MEMB(_advance)(_c_MEMB(_begin)(self), pop_idx);

    // If the element is found then remove it from the list
    if (itr.ref) 
    {
        i_key value = *(itr.ref);
        _c_MEMB(_erase_at)(self, itr); // Remove the element by value using "_erase_at".
        return value;
    }
    return *(itr.ref); // Return the element that is being popped.
}

#undef i_type
#undef i_key
#endif

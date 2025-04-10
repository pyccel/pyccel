#include <stc/priv/template.h>



// This function represents a call to the .pop() method.
// i_type: Class type (e.g., hset_int64_t).
// i_key: Data type of the elements in the set (e.g., int64_t).

static inline i_key _c_MEMB(_pull_elem)(Self* self, intptr_t pop_idx) {
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


#if defined (i_use_cmp) 
// Function to get the minimum element from the vector
static inline i_key _c_MEMB(_min)(const Self* self) {
    i_key min_val = *_c_MEMB(_front)(self);
    c_foreach(it, Self, *self) {
        if (i_less(it.ref, &min_val)) {
            min_val = *(it.ref);
        }
    }
    return min_val;
}

// Function to get the maximum element from the vector
static inline i_key _c_MEMB(_max)(const Self* self) {
    i_key max_val = *_c_MEMB(_front)(self);
    c_foreach(it, Self, *self) {
        if (i_less(&max_val, it.ref)) {
            max_val = *(it.ref);
        }
    }
    return max_val;
}

#endif

// Function to insert value before position idx
static inline void _c_MEMB(_insert)(Self* self, intptr_t idx, i_key value) {
    size_t n = _c_MEMB(_size)(self);

    if (idx < 0) idx += (intptr_t)n;
    if (idx < 0) idx = 0;

    if ((size_t)idx >= n) {
        _c_MEMB(_push_back)(self, value);
    }
    else {
        _c_MEMB(_insert_at)(self,
            _c_MEMB(_advance)(_c_MEMB(_begin)(self), idx),
            value);
    }
}

#undef i_type
#undef i_key
#undef i_use_cmp

#include <stc/priv/template2.h>

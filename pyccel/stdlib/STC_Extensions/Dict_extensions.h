#define _i_is_map
#include <stc/priv/template.h>

/**
 * Removes and returns the value associated with the key in the dictionary.
 *
 * @param self : A pointer to the dictionary instance.
 * @param rkey : The raw key to be removed from the dictionary.
 * @return The value associated with the key.
 */
static inline i_val _c_MEMB(_pop)(Self* self, i_keyraw rkey) {
    i_val value = *_c_MEMB(_at)(self, rkey);
    _c_MEMB(_erase)(self, rkey);
    return value;
}

/**
 * Removes and returns the value associated with the key in the dictionary.
 * If the key does not exist, it returns the provided default value.
 *
 * @param self        : A pointer to the dictionary instance.
 * @param rkey        : The raw key to be removed from the dictionary.
 * @param default_val : The default value to return if the key is not found.
 * @return The value associated with the key, or the default value if the key is not found.
 */
static inline i_val _c_MEMB(_pop_with_default)(Self* self, i_keyraw rkey, i_val default_val) {
    if (!_c_MEMB(_contains)(self, rkey)) {
        return default_val;
    }
    return _c_MEMB(_pop)(self, rkey);
}

/**
 * Remove and returns a (key, value) pair from the dictionary.
 *
 * @param[inout] self : A pointer to the dictionary instance.
 * @param[out] key : A key.
 * @param[out] val : The associated value.
 */
static inline void _c_MEMB(_popitem)(Self* self, i_key* key, i_val* val) {
    _m_iter iter = _c_MEMB(_begin)(self);
    *key = iter.ref->first;
    *val = iter.ref->second;
    _c_MEMB(_erase_at)(self, iter);
}

#include <stc/priv/template2.h>
#undef _i_is_map

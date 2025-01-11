#ifndef DICT_EXTENSIONS_H
#define DICT_EXTENSIONS_H

/**
 * Removes and returns the value associated with the key in the dictionary.
 *
 * @param self : A pointer to the dictionary instance.
 * @param rkey : The raw key to be removed from the dictionary.
 * @return The value associated with the key.
 */
static inline i_val _c_MEMB(_pop)(i_type* self, i_keyraw rkey) {
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
static inline i_val _c_MEMB(_pop_with_default)(i_type* self, i_keyraw rkey, i_valraw default_val) {
    if (!_c_MEMB(_contains)(self, rkey)) {
        return default_val;
    }
    return _c_MEMB(_pop)(self, rkey);
}

#undef i_type
#undef i_key
#endif // DICT_EXTENSIONS_H

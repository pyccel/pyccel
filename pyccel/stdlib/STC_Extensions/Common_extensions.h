#include <stddef.h>

static inline i_key c_JOIN(i_key,_min)(const i_key a[], size_t n) {
    i_key min_value = a[0];
    for (size_t i = 1; i < n; ++i) {
        if (a[i] < min_value) {
            min_value = a[i];
        }
    }
    return min_value;
}

static inline i_key c_JOIN(i_key, _max)(const i_key a[], size_t n) {
    i_key max_value = a[0];
    for (size_t i = 1; i < n; ++i) {
        if (a[i] > max_value) {
            max_value = a[i];
        }
    }
    return max_value;
}

#undef i_key

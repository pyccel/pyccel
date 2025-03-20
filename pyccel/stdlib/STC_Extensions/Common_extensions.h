#include <stdarg.h>

#define c_init_shared(C, ...) \
    C##_with_n_ptr(c_make_array(C##_value, __VA_ARGS__), c_sizeof((C##_value[])__VA_ARGS__)/c_sizeof(C##_value))

#ifdef i_use_cmp

static inline i_key c_JOIN(i_key,_min)(size_t count, ...) {
    va_list args;
    va_start(args, count);
    i_key min_value = va_arg(args, i_key);
    for (size_t i = 1; i < count; ++i) {
        i_key value = va_arg(args, i_key);
        if (value < min_value) {
            min_value = value;
        }
    }
    va_end(args);
    return min_value;
}

static inline i_key c_JOIN(i_key, _max)(size_t count, ...) {
    va_list args;
    va_start(args, count);
    i_key max_value = va_arg(args, i_key);
    for (size_t i = 1; i < count; ++i) {
        i_key value = va_arg(args, i_key);
        if (value > max_value) {
            max_value = value;
        }
    }
    va_end(args);
    return max_value;
}

#endif

#undef i_key

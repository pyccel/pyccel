#include <stdarg.h>

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

#undef i_key

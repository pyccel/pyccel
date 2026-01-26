#ifndef CLASS_NO_INIT_H
#define CLASS_NO_INIT_H

#include <stdint.h>

struct MethodCheck {
    int64_t my_value;
};

void stash_value(struct MethodCheck* self, int64_t val);

#endif // CLASS_NO_INIT_H

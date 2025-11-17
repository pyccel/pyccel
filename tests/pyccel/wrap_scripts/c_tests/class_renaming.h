#ifndef CLASS_PROPERTY_H
#define CLASS_PROPERTY_H

#include <stdint.h>

struct class_renaming__Counter {
    int64_t value;
};

void Counter__create(struct class_renaming__Counter* self, int64_t start);
void Counter__free(struct class_renaming__Counter* self);
void Counter__increment(struct class_renaming__Counter* self);
static inline int64_t Counter__get_value(const struct class_renaming__Counter* self) {
    return self->value;
}

#endif // CLASS_PROPERTY_H

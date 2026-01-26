#ifndef CLASS_PROPERTY_H
#define CLASS_PROPERTY_H

#include <stdint.h>

struct Counter {
    int64_t value;
};

void Counter__create(struct Counter* self, int64_t start);
void Counter__free(const struct Counter* self);
void Counter__increment(struct Counter* self);
static inline int64_t Counter__get_value(const struct Counter* self) {
    return self->value;
}

#endif // CLASS_PROPERTY_H

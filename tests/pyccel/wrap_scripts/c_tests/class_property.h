#ifndef CLASS_PROPERTY_H
#define CLASS_PROPERTY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

struct Counter {
    int64_t value;
};

void Counter__create(struct Counter* self, int64_t start);
void Counter__free(const struct Counter* self);
void Counter__increment(struct Counter* self);
int64_t Counter__get_value(const struct Counter* self);

#endif // class_property_H

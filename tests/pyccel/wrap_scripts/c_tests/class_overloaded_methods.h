#ifndef CLASS_OVERLOADED_METHODS_H
#define CLASS_OVERLOADED_METHODS_H

#include <stdint.h>

struct Adder {
};

void Adder__create(struct Adder* self);
void Adder__free(struct Adder* self);
int64_t Adder__add_0000(struct Adder* self, int64_t x, int64_t y);
double Adder__add_0001(struct Adder* self, double x, double y);

#endif // CLASS_OVERLOADED_METHODS_H

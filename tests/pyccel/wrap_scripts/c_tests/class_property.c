#include "class_property.h"


/*........................................*/
void Counter__create(struct Counter* self, int64_t start)
{
    self->value = start;
}
/*........................................*/
/*........................................*/
void Counter__free(const struct Counter* self)
{
}
/*........................................*/
/*........................................*/
void Counter__increment(struct Counter* self)
{
    self->value += INT64_C(1);
}
/*........................................*/

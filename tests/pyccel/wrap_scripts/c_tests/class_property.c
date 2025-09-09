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
/*........................................*/
int64_t Counter__get_value(const struct Counter* self)
{
    return self->value;
}
/*........................................*/

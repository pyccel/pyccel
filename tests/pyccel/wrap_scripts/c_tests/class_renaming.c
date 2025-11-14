#include "class_renaming.h"


/*........................................*/
void Counter__create(struct class_renaming__Counter* self, int64_t start)
{
    self->value = start;
}
/*........................................*/
/*........................................*/
void Counter__free(const struct class_renaming__Counter* self)
{
    self->value = INT64_C(-1);
}
/*........................................*/
/*........................................*/
void Counter__increment(struct class_renaming__Counter* self)
{
    self->value += INT64_C(1);
}
/*........................................*/

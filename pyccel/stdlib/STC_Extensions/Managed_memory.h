
#include <stc/priv/linkage.h>

#ifndef PYCCEL_MANAGED_MEM_H_INCLUDED
#define PYCCEL_MANAGED_MEM_H_INCLUDED
#include <stc/common.h>

#ifndef PYCCEL_TYPES_H_INCLUDED
#define PYCCEL_TYPES_H_INCLUDED

#define _c_mgd_types(SELF, VAL) \
    typedef VAL SELF##_value; \
    typedef struct SELF { \
        SELF##_value* get; \
        bool is_owning; \
    } SELF
#endif // PYCCEL_TYPES_H_INCLUDED

#include <stdlib.h>
#endif // PYCCEL_MANAGED_MEM_H_INCLUDED

#define _i_is_arc
#include <stc/priv/template.h>
#ifndef i_declared
_c_DEFTYPES(_c_mgd_types, Self, i_key);
#endif

STC_INLINE Self _c_MEMB(_init)(void)
    { return c_literal(Self){false, NULL}; }

STC_INLINE Self _c_MEMB(_make)(_m_value val) {
    Self owned;
    owned.is_owning = true;
    owned.get = _i_malloc(_m_value, 1);
    *owned.get = val;
    return owned;
}

STC_INLINE Self _c_MEMB(_from_ptr)(_m_value* ptr) {
    Self unowned;
    unowned.is_owning = false;
    unowned.get = ptr;
    return unowned;
}

STC_INLINE void _c_MEMB(_drop)(const Self* self) {
    if (self->is_owning) {
        i_keydrop(self->get);
        i_free(self->get, c_sizeof *self->get);
    }
}


#include <stc/priv/linkage2.h>
#include <stc/priv/template2.h>

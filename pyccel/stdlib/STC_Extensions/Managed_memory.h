#include <stc/priv/linkage.h>

#ifndef PYCCEL_MANAGED_MEM_H_INCLUDED
#define PYCCEL_MANAGED_MEM_H_INCLUDED
#include <stc/common.h>

#ifndef PYCCEL_TYPES_H_INCLUDED
#define PYCCEL_TYPES_H_INCLUDED

#define _c_mgd_types(SELF, VAL) \
    typedef VAL SELF##_value;   \
    typedef struct SELF {       \
        SELF##_value* get;      \
        bool is_owning;         \
    } SELF
#endif  // PYCCEL_TYPES_H_INCLUDED

#include <stdlib.h>
#endif  // PYCCEL_MANAGED_MEM_H_INCLUDED

#define _i_is_arc
#include <stc/priv/template.h>
#ifndef i_declared
_c_DEFTYPES(_c_mgd_types, Self, i_key);
#endif

STC_INLINE Self _c_MEMB(_init)(void) { return c_literal(Self){false, NULL}; }

STC_INLINE Self _c_MEMB(_make)(_m_value val) {
    Self owned;
    owned.is_owning = true;
    owned.get = i_malloc(c_sizeof(_m_value));
    *owned.get = val;
    return owned;
}

STC_INLINE Self _c_MEMB(_from_ptr)(_m_value* ptr) {
    Self unowned;
    unowned.is_owning = false;
    unowned.get = ptr;
    return unowned;
}

STC_INLINE Self _c_MEMB(_clone)(const Self self) {
    return _c_MEMB(_from_ptr)(self.get);
}

STC_INLINE Self _c_MEMB(_steal)(const Self self) {
    return self;
}

STC_INLINE void _c_MEMB(_drop)(const Self* self) {
    if (self->is_owning) {
        i_keydrop(self->get);
        i_free(self->get, c_sizeof(*self->get));
    }
}

STC_INLINE _m_value* _c_MEMB(_take_ptr)(Self self) {
    _m_value* out = self.get;
    return out;
}

STC_INLINE _m_value _c_MEMB(_release)(Self self) {
    c_assert(self.is_owning);
    _m_value out = *self.get;
    i_free(self.get, c_sizeof(*self.get));
    return out;
}

#undef _i_is_arc
#include <stc/priv/linkage2.h>
#include <stc/priv/template2.h>

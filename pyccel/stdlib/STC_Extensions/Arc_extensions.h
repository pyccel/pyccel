#define _i_is_arc
#include <stc/priv/linkage.h>
#include <stc/priv/template.h>

STC_INLINE Self _c_MEMB(_from_heap)(i_key raw)
{ return _c_MEMB(_make)(i_keyfrom(raw));
}

STC_INLINE c_JOIN(i_key, _value) _c_MEMB(_release)(Self mem) {
    c_JOIN(i_key, _value)* out_ptr = mem.get->get;
    c_assert(mem.get->is_owning);
    mem.get->get = NULL;
    _c_MEMB(_drop)(&mem);
    c_JOIN(i_key, _value) out = *out_ptr;
    i_free(out_ptr, c_sizeof out);
    return out;
}

#include <stc/priv/template2.h>
#include <stc/priv/linkage2.h>

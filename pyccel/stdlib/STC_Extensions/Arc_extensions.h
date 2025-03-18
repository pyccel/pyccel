#define _i_is_arc
#include <stc/priv/template.h>

struct {
    bool is_owning;
    i_key* ptr;
};

static inline Self _c_MEMB(_share)(i_keyraw x) {
    Self out = _c_MEMB(_make)(x);
    (*out.use_count)++;
    return out;
}

#include <stc/priv/template2.h>

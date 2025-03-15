#define _i_is_arc
#include <stc/priv/template.h>

static inline Self _c_MEMB(_share)(i_keyraw x) {
    element_vec_vec_int64_t out = element_vec_vec_int64_t_make(x);
    (*out.use_count)++;
    return out;
}

#include <stc/priv/template2.h>

#ifndef SET_EXTENSIONS_H
#define SET_EXTENSIONS_H


#define _c_MEMB(name) c_JOIN(i_type, name)

static inline i_key _c_MEMB(_pop)(i_type* self) {
    _c_MEMB(_iter) itr = _c_MEMB(_begin)(self);
    if (itr.ref)
    {
        i_key value = *(itr.ref);
        _c_MEMB(_erase_at)(self, itr);
        return value;
    }
    return *(itr.ref);
}

#endif

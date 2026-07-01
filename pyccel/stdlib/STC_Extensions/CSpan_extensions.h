#ifndef CSPAN_EXTENSIONS_H
#define CSPAN_EXTENSIONS_H

#define cspan_copy(target_type, Span_lhs, Span_rhs, lhs, rhs) \
    { \
        Span_lhs##_iter it_lhs; \
        Span_rhs##_iter it_rhs; \
        for (it_lhs = Span_lhs##_begin(lhs), it_rhs = Span_rhs##_begin(rhs); \
                it_lhs.ref && it_rhs.ref; \
                Span_lhs##_next(&it_lhs), Span_rhs##_next(&it_rhs)) { \
            *(it_lhs.ref) = (target_type)(*(it_rhs.ref)); \
        } \
    }

#endif

#define contig_cspan_at(self, ...) ((self)->data + contig_cspan_index(self, __VA_ARGS__))

#define contig_cspan_index(...) cspan_index_fn(__VA_ARGS__, c_COMMA_N(cspan_index_3d), c_COMMA_N(cspan_index_2d), \
                                                     c_COMMA_N(contig_cspan_index_1d),)(__VA_ARGS__)
#define contig_cspan_index_1d(self, i)     (c_static_assert(cspan_rank(self) == 1), \
                                     c_assert((i) < (self)->shape[0]), \
                                     i)

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

#define contig_cspan_index(...) contig_cspan_index_fn(__VA_ARGS__, c_COMMA_N(contig_cspan_index_3d), c_COMMA_N(contig_cspan_index_2d), \
                                                     c_COMMA_N(contig_cspan_index_1d),)(__VA_ARGS__)
#define contig_cspan_index_fn(self, i,j,k,n, ...) c_TUPLE_AT_1(n, contig_cspan_index_nd,)
#define contig_cspan_index_1d(self, i)     (c_static_assert(cspan_rank(self) == 1), \
                                     c_assert((i) < (self)->shape[0]), \
                                     i)
#define contig_cspan_index_2d(self, i,j)   (c_static_assert(cspan_rank(self) == 2), \
                                     c_assert((i) < (self)->shape[0] && (j) < (self)->shape[1]), \
                                     (i)*(self)->stride.d[0] + j)
#define contig_cspan_index_3d(self, i,j,k) (c_static_assert(cspan_rank(self) == 3), \
                                     c_assert((i) < (self)->shape[0] && (j) < (self)->shape[1] && (k) < (self)->shape[2]), \
                                     (i)*(self)->stride.d[0] + (j)*(self)->stride.d[1] + k)
#define contig_cspan_index_nd(self, ...) _cspan_index((self)->shape, (self)->stride.d, c_make_array(isize, {__VA_ARGS__}), \
                                               (c_static_assert(cspan_rank(self) == c_NUMARGS(__VA_ARGS__)), cspan_rank(self)))


#define cspan_contig_index_1d(self, i)  (c_static_assert(cspan_rank(self) == 1), \
                                         c_assert((i) < (self)->shape[0]), \
                                         i)

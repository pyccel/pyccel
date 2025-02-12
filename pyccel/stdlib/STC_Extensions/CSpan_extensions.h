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

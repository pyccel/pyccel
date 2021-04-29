#define APPLY1(t,n, a) t(n, a)
#define APPLY2(t,n, a, b) APPLY1(t, n - 1, a) + t(n, b)
#define APPLY3(t,n, a, b, c) APPLY2(t, n - 1, a, b) + t(n, c)
#define APPLY4(t,n, a, b, c, d) APPLY3(t, n - 1, a, b, c) + t(n, d)
#define APPLY5(t,n, a, b, c, d, e) APPLY4(t, n - 1, a, b, c, d) + t(n, e)
#define APPLY6(t,n, a, b, c, d, e, f) APPLY5(t, n - 1, a, b, c, d, e) + t(n, f)

#define NUM_ARGS_H1(dummy, x6, x5, x4, x3, x2, x1, x0, ...) x0
#define NUM_ARGS(...) NUM_ARGS_H1(dummy, __VA_ARGS__, 6, 5, 4, 3, 2, 1, 0)
#define APPLY_ALL_H3(t, n, ...) APPLY##n(t, n - 1, __VA_ARGS__)
#define APPLY_ALL_H2(t, n, ...) APPLY_ALL_H3(t, n, __VA_ARGS__)
#define APPLY_ALL(t, ...) APPLY_ALL_H2(t, NUM_ARGS(__VA_ARGS__), __VA_ARGS__)

#define GET_INDEX(...) APPLY_ALL(INDEX, __VA_ARGS__)
#define INDEX(n, a) ([n] * a)
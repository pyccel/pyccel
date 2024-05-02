#ifndef NDARRAYS_H
# define NDARRAYS_H

# include <stdlib.h>
# include <complex.h>
# include <string.h>
# include <stdio.h>
# include <stdarg.h>
# include <stdbool.h>

# define array_fill(c, arr) _Generic((c), int : _array_fill_int,\
                                        float : _array_fill_float,\
                                        double : _array_fill_double,\
                                        double complex : _array_fill_cdouble)(c, arr)
typedef struct  s_slice
{
    int start;
    int end;
    int step;
}               t_slice;

enum e_types
{
        nd_int,
        nd_float,
        nd_double,
        nd_cdouble
};

typedef struct  s_ndarray
{
    /* raw data buffer*/
    union {
            char            *raw_data;
            int             *nd_int;
            float           *nd_float;
            double          *nd_double;
            double complex  *nd_cdouble;
            };
    /* number of dimensions */
    int             nd;
    /* shape 'size of each dimension' */
    int             *shape;
    /* strides 'number of bytes to skip to get the next element' */
    int             *strides;
    /* type of the array elements */
    enum e_types    type;
    int             type_size;
    int             length;
    int             buffer_size;
    bool            is_slice;
}               t_ndarray;

/* functions prototypes */

/* allocations */
t_ndarray   array_create(int nd, int *shape, enum e_types type);
void        _array_fill_int(int c, t_ndarray arr);
void        _array_fill_float(float c, t_ndarray arr);
void        _array_fill_double(double c, t_ndarray arr);
void        _array_fill_cdouble(complex double c, t_ndarray arr);

/* slicing */
t_slice     new_slice(int start, int end, int step);
t_ndarray   array_slicing(t_ndarray p, ...);

/* free */
int         free_array(t_ndarray dump);

/* indexing */
int         get_index(t_ndarray arr, ...);

#endif

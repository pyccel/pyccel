#ifndef NDARRAYS_H
# define NDARRAYS_H

# include <stdlib.h>
# include <complex.h>
# include <string.h>
# include <stdio.h>
# include <stdarg.h>
# include <stdbool.h>
# include <stdint.h>

/* mapping the function array_fill to the correct type */
# define array_fill(c, arr) _Generic((c), int64_t : _array_fill_int64,\
                                        int32_t : _array_fill_int32,\
                                        int16_t : _array_fill_int16,\
                                        int8_t : _array_fill_int8,\
                                        float : _array_fill_float,\
                                        double : _array_fill_double,\
                                        bool : _array_fill_bool,\
                                        float complex : _array_fill_cfloat,\
                                        double complex : _array_fill_cdouble)(c, arr)
typedef struct  s_slice
{
    int start;
    int end;
    int step;
}               t_slice;

enum e_types
{
        nd_bool,
        nd_int8,
        nd_int16,
        nd_int32,
        nd_int64,
        nd_float,
        nd_double,
        nd_cfloat,
        nd_cdouble
};

typedef struct  s_ndarray
{
    /* raw data buffer*/
    union {
            char            *raw_data;
            int8_t          *nd_int8;
            int16_t         *nd_int16;
            int32_t         *nd_int32;
            int64_t         *nd_int64;
            float           *nd_float;
            double          *nd_double;
            bool            *nd_bool;
            double complex  *nd_cdouble;
            float  complex  *nd_cfloat;
            };
    /* number of dimensions */
    int                 nd;
    /* shape 'size of each dimension' */
    int                 *shape;
    /* strides 'number of bytes to skip to get the next element' */
    int                 *strides;
    /* type of the array elements */
    enum e_types        type;
    /* type size of the array elements */
    int                 type_size;
    /* number of element in the array */
    int                 length;
    /* size of the array */
    int                 buffer_size;
    bool                is_slice;
}               t_ndarray;

/* functions prototypes */

/* allocations */
t_ndarray   array_create(int nd, int *shape, enum e_types type);
void        _array_fill_int8(int8_t c, t_ndarray arr);
void        _array_fill_int16(int16_t c, t_ndarray arr);
void        _array_fill_int32(int32_t c, t_ndarray arr);
void        _array_fill_int64(int64_t c, t_ndarray arr);
void        _array_fill_float(float c, t_ndarray arr);
void        _array_fill_double(double c, t_ndarray arr);
void        _array_fill_bool(bool c, t_ndarray arr);
void        _array_fill_cfloat(float complex c, t_ndarray arr);
void        _array_fill_cdouble(double complex c, t_ndarray arr);

/* slicing */
                /* creating a Slice object */
t_slice     new_slice(int start, int end, int step);
                /* creating an array view */
t_ndarray   array_slicing(t_ndarray p, ...);

/* free */
int         free_array(t_ndarray dump);

/* indexing */
int         get_index(t_ndarray arr, ...);

#endif

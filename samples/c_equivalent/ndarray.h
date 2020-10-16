#ifndef NDARRAY_H
# define NDAARAY_H

# define NDARRAY_MAX_DIMS 32

typedef union s_ndarr_type
{
    /* data --- change it later to shorter names*/
    char *raw_data;
    int *int_nd;
    float *float_nd;
    double *double_nd;

}  t_ndarray_type;

typedef struct s_slice_data{
    int start;
    int end;
    int step;
}  t_slice;

typedef struct s_ndarray
{
    /* raw data buffer*/
    t_ndarray_type *data;
    /* number of dimmensions */
    int nd;
    /* shape 'size of each dimmension' */
    int *shape;
    /* strides 'number of bytes to skip to get the next element' */
    int *strides;
    /* type of the array elements */
    int type; // TODO : make it into an enum
    int lenght;
    int is_slice;
} t_ndarray;

#endif
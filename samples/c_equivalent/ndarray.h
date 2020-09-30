#ifndef NDARRAY_H
# define NDAARAY_H

# define NDARRAY_MAX_DIMS 32

typedef union s_ndarr_type
{
    /* data */
    char *raw_data;
    int *int_nd;
    float *float_nd;
    double *double_nd;

}  t_ndarray_type;

typedef struct s_ndarray
{
    /* raw data buffer*/
    t_ndarray_type buffer;
    /* number of dimmensions */
    int nd;
    /* shape 'size of each dimmension' */
    int shape[NDARRAY_MAX_DIMS];
    /* type of the array elements */
    int types; // TODO : make it into an enum

} t_ndarray;

#endif
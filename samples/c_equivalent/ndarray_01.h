#ifndef NDARRAY01_H
# define NDAARAY01_H

# define NDARRAY_MAX_DIMS 32

typedef struct s_ndarray
{
    /* raw data buffer*/
    char *data;
    /* number of dimmensions */
    int nd;
    /* shape 'size of each dimmension' */
    int *shape;
    /* type of the array elements */
    int type; // TODO : make it into an enum
    /* strides */
    int *strides;

} t_ndarray;

#endif
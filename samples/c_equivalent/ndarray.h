typedef struct s_ndarray
{
    /* raw data buffer*/
    char *data;
    /* number of dimmensions */
    int nd;
    /* shape 'size of each dimmension' */

    /* type of the array elements */
    int type;


}t_ndarray;

typedef union s_ndarr_type
{
    /* data */
    char *raw_data;
    int *int_nd;
    float *float_nd;
    double *double_nd;

}  t_ndarray_type;

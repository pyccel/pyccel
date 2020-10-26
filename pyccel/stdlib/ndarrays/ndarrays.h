#ifndef NDARRAYS_H
# define NDARRAYS_H

# include <stdlib.h>
# include <complex.h>
# include <string.h>
# include <stdio.h>
# include <stdarg.h>

typedef union	u_ndarr_type
{
	/*
	*** data --- change it later to shorter names or use anonymous array
	*/
	char	*raw_data;
	int		*int_nd;
	float	*float_nd;
	double	*double_nd;
    double complex *complex_double;
}				t_ndarray_type;

typedef struct s_slice_data{
	int start;
	int end;
	int step;
}	t_slice;

typedef struct s_ndarray
{
	/* raw data buffer*/
	t_ndarray_type *data;
	/* number of dimmensions */
	int	nd;
	/* shape 'size of each dimmension' */
	int *shape;
	/* strides 'number of bytes to skip to get the next element' */
	int *strides;
	/* type of the array elements */
	int type; // TODO : make it into an enum
	int lenght;
	int is_slice;
} t_ndarray;

/* functions prototypes */

/* allocations */
t_ndarray *init_array(char *temp, int nd, int *shape, int type);

/* dumping data */
// int array_value_dump(t_ndarray *arr);
// int array_data_dump(t_ndarray *arr);

/* slicing */
t_slice *slice_data(int start, int end, int step);
t_ndarray *make_slice(t_ndarray *p, ...);

/* free */
int free_array(t_ndarray *dump);

/* indexing */
int get_index(t_ndarray *arr, ...);

/* other funcs */
t_ndarray *mat_product(t_ndarray *mat1, t_ndarray *mat2);

#endif

# Multidimensional ndarrays memory layout

`ndarrays` are stored as contiguous data in memory to increase efficiency
We use `copy_array_data(t_ndarray *dest, t_ndarray *src, t_uint32 offset)` to achieve this wether it `order_f` or `order_c`.

`order_c` is a row-major order, where we store data contiguously row by row
`order_f` is a column-major order, where we store data contiguously column by column

Taking as an example: `ab = array([a, b])` where `a = array([1, 2, 3])` and `b = array([4, 5, 6])`
`ab` would be stored like this:
![C order memory layout](media/c_order_memory_layout.png)



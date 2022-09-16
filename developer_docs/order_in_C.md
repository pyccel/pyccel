# Multidimensional ndarrays memory layout

## Order

### Ordering in C

### Ordering in Fortran

`ndarrays` are stored as contiguous data in memory to increase efficiency
We use `copy_array_data(t_ndarray *dest, t_ndarray *src, t_uint32 offset, enum order)` to achieve this whether the order is `order_f` or `order_c`.

Taking as an example: `ab = array([a, b])` where `a = array([1, 2, 3])` and `b = array([4, 5, 6])`  

`ab` would be stored like this in `order_c`:


![C order memory layout](media/c_order_memory_layout.png)  

and like this in `order_f`:  

![F order memory layout](media/f_order_memory_layout.png)  

## Calls to `array_copy_data`

Taking as an example: `ab = array([a, b])` where `a = array([1, 2, 3])` and `b = array([4, 5, 6])`  

For `order_c`:
after allocating space for the ndarray `ab` (and supposing that arrays `a` and `b` have already been created), we would call:
  1. ```array_copy_data(ab, a, 0, order_c)```, this will copy `a` to `ab` starting from `index=0` + `offset`, to `index=size-of(a)` + `offset`, and since `offset` is 0, `a` will be copied starting from `index=0` to `index=3`, the result is:  

![C order memory layout](media/first_order_c_array_copy.png)

  2. ```array_copy_data(ab, b, order_c)```, this will copy `b` to `ab` starting from `index=0` + `offset`, to `index=size-of(b)` + `offset`, and since `offset` is 3 (size-of(b)), `a` will be copied to `ab` starting from `index=3` to `index=6`, the result is:  

![C order memory layout](media/c_order_memory_layout.png)

For `order_f`:
after allocating space for the ndarray `ab` (and supposing that arrays `a` and `b` have already been created), we would call:  
 1. ```array_copy_data(ab, a, 0, order_f)```, this will copy N elements of `a` (N=size-of(`a`)) into their correct memory position using strides and shape, so it will be copying one element to `index=0` + `offset`, but jumping one to copy the next element to `index=2` + `offset`, the result is:  
 
 ![F order memory layout](media/first_order_f_array_copy.png)

 2. ```array_copy_data(ab, b, 1, order_f)```, This will do the same for `b`, but since the `offset` is 1, the copying will start from `index=0` + 1, this will result in the final array:  

 ![F order memory layout](media/f_order_memory_layout.png) 
 
 ## Nested arrays handling (for now)

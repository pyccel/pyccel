# Supported Numpy function by Pyccel

In Pyccel we try to support the Numpy functions which developers use the most.. Here are some of them:

## [norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

-   Supported parameters:

    x: array_like
       Input array. If axis is None, x must be 1-D or 2-D, unless ord is None.
       If both axis and ord are None, the 2-norm of x.ravel will be returned.

    axis: {None, int, 2-tuple of ints}, optional.
         If axis is an integer, it specifies the axis of x along which to compute the vector norms.
         If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of
         these matrices are computed. If axis is None then either a vector norm (when x is 1-D) or a
         matrix norm (when x is 2-D) is returned. The default is None. New in version 1.8.0.

-   Supported languages: Fortran (2-norm)

-   python code:

    ```python
    from numpy.linalg import norm
    from numpy import array
    arr1 = array([1,2,3,4])
    nrm = norm(arr1)
    print(nrm)

    arr2 = array([[1,2,3,4],[4,3,2,1]])
    nrm2 = norm(arr2, axis=1)
    print(nrm)
    ```

-   fortran equivalent:

    ```fortran
    program prog_test_norm

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(C_INT64_T), allocatable :: arr1(:)
    real(C_DOUBLE) :: nrm
    integer(C_INT64_T), allocatable :: arr2(:,:)
    real(C_DOUBLE), allocatable :: nrm2(:)

    allocate(arr1(0:3_C_INT64_T))
    arr1 = [1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T]
    nrm = Norm2(Real(arr1, C_DOUBLE))
    print *, nrm
    allocate(arr2(0:3_C_INT64_T, 0:1_C_INT64_T))
    arr2 = reshape([[1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T], [ &
      4_C_INT64_T, 3_C_INT64_T, 2_C_INT64_T, 1_C_INT64_T]], [ &
      4_C_INT64_T, 2_C_INT64_T])
    allocate(nrm2(0:1_C_INT64_T))
    nrm2 = Norm2(Real(arr2, C_DOUBLE),1_C_INT64_T)
    print *, nrm

    end program prog_test_norm
    ```

## [real](https://numpy.org/doc/stable/reference/generated/numpy.real.html) and [imag](https://numpy.org/doc/stable/reference/generated/numpy.imag.html) functions

-   Supported languages: C, fortran

-   python code:

    ```python
    from numpy import imag, real, array
    arr1 = array([1+1j,2+1j,3+1j,4+1j])
    real_part = real(arr1)
    imag_part = imag(arr1)
    print("real part for arr1: " , real_part, "\nimag part for arr1: ", imag_part)
    ```

-   fortran equivalent:

    ```fortran
    program prog_test_imag_real

    use, intrinsic :: ISO_C_BINDING

    implicit none

    complex(C_DOUBLE_COMPLEX), allocatable :: arr1(:)
    real(C_DOUBLE), allocatable :: real_part(:)
    real(C_DOUBLE), allocatable :: imag_part(:)

    allocate(arr1(0:3_C_INT64_T))
    arr1 = [(1.0_C_DOUBLE, 1.0_C_DOUBLE), (2.0_C_DOUBLE, 1.0_C_DOUBLE), ( &
          3.0_C_DOUBLE, 1.0_C_DOUBLE), (4.0_C_DOUBLE, 1.0_C_DOUBLE)]
    allocate(real_part(0:3_C_INT64_T))
    real_part = Real(arr1, C_DOUBLE)
    allocate(imag_part(0:3_C_INT64_T))
    imag_part = aimag(arr1)
    print *, 'real part for arr1: ' // ' ' , real_part, ACHAR(10) // 'imag part for arr1: ' // ' ' , imag_part

    end program prog_test_imag_real
    ```

-   C equivalent:

    ```C
    #include <complex.h>
    #include <stdlib.h>
    #include "ndarrays.h"
    #include <stdio.h>
    #include <stdint.h>
    int main()
    {
        t_ndarray arr1;
        t_ndarray real_part;
        t_ndarray imag_part;
        int64_t i_0001;
        int64_t i;
        int64_t i_0002;
        arr1 = array_create(1, (int64_t[]){4}, nd_cdouble);
        double complex array_dummy_0001[] = {(1.0 + 1.0 * _Complex_I), (2.0 + 1.0 * _Complex_I), (3.0 + 1.0 * _Complex_I), (4.0 + 1.0 * _Complex_I)};
        memcpy(arr1.nd_cdouble, array_dummy_0001, arr1.buffer_size);
        real_part = array_create(1, (int64_t[]){4}, nd_double);
        for (i_0001 = 0; i_0001 < 4; i_0001 += 1)
        {
            real_part.nd_double[get_index(real_part, i_0001)] = creal(arr1.nd_cdouble[get_index(arr1, i_0001)]);
        }
        imag_part = array_create(1, (int64_t[]){4}, nd_double);
        for (i_0001 = 0; i_0001 < 4; i_0001 += 1)
        {
            imag_part.nd_double[get_index(imag_part, i_0001)] = cimag(arr1.nd_cdouble[get_index(arr1, i_0001)]);
        }
        printf("%s ", "real part for arr1: ");
        printf("%s", "[");
        for (i = 0; i < 3; i += 1)
        {
            printf("%.12lf ", real_part.nd_double[get_index(real_part, i)]);
        }
        printf("%.12lf]", real_part.nd_double[get_index(real_part, 3)]);
        printf("%s ", "\nimag part for arr1: ");
        printf("%s", "[");
        for (i_0002 = 0; i_0002 < 3; i_0002 += 1)
        {
            printf("%.12lf ", imag_part.nd_double[get_index(imag_part, i_0002)]);
        }
        printf("%.12lf]\n", imag_part.nd_double[get_index(imag_part, 3)]);
        free_array(arr1);
        free_array(real_part);
        free_array(imag_part);
        return 0;
    }
    ```

-   python code with arrays:

    ```python
    from numpy import imag, real, array
    arr1 = array([1+1j,2+1j,3+1j,4+1j])
    real_part = real(arr1)
    imag_part = imag(arr1)
    print("real part for arr1: " , real_part, "\nimag part for arr1: ", imag_part)
    ```

-   fortran equivalent:

    ```fortran
    program prog_test_imag_real

    use, intrinsic :: ISO_C_BINDING

    implicit none

    complex(C_DOUBLE_COMPLEX), allocatable :: arr1(:)
    real(C_DOUBLE), allocatable :: real_part(:)
    real(C_DOUBLE), allocatable :: imag_part(:)

    allocate(arr1(0:3_C_INT64_T))
    arr1 = [(1.0_C_DOUBLE, 1.0_C_DOUBLE), (2.0_C_DOUBLE, 1.0_C_DOUBLE), ( &
          3.0_C_DOUBLE, 1.0_C_DOUBLE), (4.0_C_DOUBLE, 1.0_C_DOUBLE)]
    allocate(real_part(0:3_C_INT64_T))
    real_part = Real(arr1, C_DOUBLE)
    allocate(imag_part(0:3_C_INT64_T))
    imag_part = aimag(arr1)
    print *, 'real part for arr1: ' // ' ' , real_part, ACHAR(10) // 'imag part for arr1: ' // ' ' , imag_part

    end program prog_test_imag_real
    ```

-   C equivalent:

    ```C
    #include <stdlib.h>
    #include <stdio.h>
    #include "ndarrays.h"
    #include <complex.h>
    #include <stdint.h>
    int main()
    {
        t_ndarray arr1;
        t_ndarray real_part;
        t_ndarray imag_part;
        int64_t i_0001;
        int64_t i;
        int64_t i_0002;
        arr1 = array_create(1, (int64_t[]){4}, nd_cdouble);
        double complex array_dummy_0001[] = {(1.0 + 1.0 * _Complex_I), (2.0 + 1.0 * _Complex_I), (3.0 + 1.0 * _Complex_I), (4.0 + 1.0 * _Complex_I)};
        memcpy(arr1.nd_cdouble, array_dummy_0001, arr1.buffer_size);
        real_part = array_create(1, (int64_t[]){4}, nd_double);
        for (i_0001 = 0; i_0001 < 4; i_0001 += 1)
        {
            real_part.nd_double[get_index(real_part, i_0001)] = creal(arr1.nd_cdouble[get_index(arr1, i_0001)]);
        }
        imag_part = array_create(1, (int64_t[]){4}, nd_double);
        for (i_0001 = 0; i_0001 < 4; i_0001 += 1)
        {
            imag_part.nd_double[get_index(imag_part, i_0001)] = cimag(arr1.nd_cdouble[get_index(arr1, i_0001)]);
        }
        printf("%s ", "real part for arr1: ");
        printf("%s", "[");
        for (i = 0; i < 4 - 1; i += 1)
        {
            printf("%.12lf ", real_part.nd_double[get_index(real_part, i)]);
        }
        printf("%.12lf]", real_part.nd_double[get_index(real_part, 4 - 1)]);
        printf("%s ", "\nimag part for arr1: ");
        printf("%s", "[");
        for (i_0002 = 0; i_0002 < 4 - 1; i_0002 += 1)
        {
            printf("%.12lf ", imag_part.nd_double[get_index(imag_part, i_0002)]);
        }
        printf("%.12lf]\n", imag_part.nd_double[get_index(imag_part, 4 - 1)]);
        free_array(arr1);
        free_array(real_part);
        free_array(imag_part);
        return 0;
    }
    ```

## [prod](https://numpy.org/doc/stable/reference/generated/numpy.prod.html)

-   Supported parameters:

    a: array_like,
        Input data.

-   Supported languages: fortran

-   python code:

    ```python
    from numpy import array, prod

    arr = array([1,2,3,4])
    prd = prod(arr)
    print("prd: ", prd)
    ```

-   fortran equivalent:

    ```fortran
    program prog_test_prod

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(C_INT64_T), allocatable :: arr(:)
    integer(C_INT64_T) :: prd

    allocate(arr(0:3_C_INT64_T))
    arr = [1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T]
    prd = product(arr)
    print *, 'prd: ' // ' ' , prd

    end program prog_test_prod
    ```

## [mod](https://numpy.org/doc/stable/reference/generated/numpy.mod.html)

-   Supported parameters:

    x1: array_like
        Dividend array.

    x2: array_like,
        Divisor array. If x1.shape != x2.shape, they must be
        broadcastable to a common shape (which becomes the shape of the output).

-   Supported language: fortran.

-   python code:

    ```python
    from numpy import array, mod

    arr = array([1,2,3,4])
    res = mod(arr, arr)
    print("res: ", res)
    ```

-   fortran equivalent:

    ```fortran
    program prog_test_mod

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(C_INT64_T), allocatable :: arr(:)
    integer(C_INT64_T), allocatable :: res(:)

    allocate(arr(0:3_C_INT64_T))
    arr = [1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T]
    allocate(res(0:3_C_INT64_T))
    res = MODULO(arr,arr)
    print *, 'res: ' // ' ' , res

    end program prog_test_prod
    ```

## [matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)

-   Supported parameters:

    x1, x2: array_like,
        Input arrays (must be 1d or 2d), scalars not allowed.

-   Supported languages: fortran (1d or 2d arrays only).

-   python code:

    ```python
    from numpy import array, matmul

    arr = array([[1,2],[3,4]])
    res = matmul(arr, arr)
    print("res: ", res)
    ```

-   fortran equivalent:

    ```fortran
    program prog_test_matmul

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(C_INT64_T), allocatable :: arr(:,:)
    integer(C_INT64_T), allocatable :: res(:,:)

    allocate(arr(0:1_C_INT64_T, 0:1_C_INT64_T))
    arr = reshape([[1_C_INT64_T, 2_C_INT64_T], [3_C_INT64_T, 4_C_INT64_T]], &
          [2_C_INT64_T, 2_C_INT64_T])
    allocate(res(0:1_C_INT64_T, 0:1_C_INT64_T))
    res = matmul(arr,arr)
    print *, 'res: ' // ' ' , res

    end program prog_test_matmul
    ```

## [linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

-   Supported languages: C, fortran

-   Supported parameters:

    start, stop: array_like,

    num: int, optional (Default is 50)

    endpoint: bool, optional (Default is True)

    dtype: dtype, optional

-   python code:

    ```python
    from numpy import linspace

    if __name__ == "__main__":
        x = linspace(0, 10, 20, endpoint=True, dtype='float64')
        print(x)
    ```

-   fortran equivalent:

    ```fortran
    program prog_prog_test

      use test

      use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f64 => &
          C_DOUBLE
      implicit none

      real(f64), allocatable :: x(:)
      integer(i64) :: linspace_index

      allocate(x(0:19_i64))
      x = [((0_i64 + linspace_index*Real((10_i64 - 0_i64), f64) / Real(( &
          20_i64 - 1_i64), f64)), linspace_index = 0_i64,19_i64)]
      x(19_i64) = 10.0_f64
      print *, x
      if (allocated(x)) then
        deallocate(x)
      end if

    end program prog_prog_test
    ```

-   C equivalent:

    ```C
    #include "test.h"
    #include "ndarrays.h"
    #include <stdlib.h>
    #include <stdint.h>
    #include <stdio.h>
    int main()
    {
        t_ndarray x;
        int64_t i_0001;
        int64_t i;
        x = array_create(1, (int64_t[]){20}, nd_double);
        for (i_0001 = 0; i_0001 < 20; i_0001 += 1)
        {
            GET_ELEMENT(x, nd_double, i_0001) = (0 + i_0001*(double)((10 - 0)) / (double)((20 - 1)));
            GET_ELEMENT(x, nd_double, 19) = (double)10;
        }
        printf("%s", "[");
        for (i = 0; i < 19; i += 1)
        {
            printf("%.12lf ", GET_ELEMENT(x, nd_double, i));
        }
        printf("%.12lf]\n", GET_ELEMENT(x, nd_double, 19));
        free_array(x);
        return 0;
    }
    ```

## [Transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)

-   Supported languages: C, fortran

-   Supported parameters:

    a: array_like,

-   python code:

    ```python
    from numpy import transpose

    def print_transpose(y : 'int[:,:,:]'):
        print(transpose(y))
        print(y.T)
    ```

-   fortran equivalent:

    ```fortran
    program prog_prog_tmp

      use tmp

      use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f64 => &
            C_DOUBLE
      implicit none

      real(f64), allocatable :: a(:,:)
      integer(i64) :: i
      integer(i64) :: j
      real(f64), allocatable :: b(:,:)
      real(f64), allocatable :: c(:,:)
      integer(i64) :: i_0001

      allocate(a(0:3_i64, 0:2_i64))
      do i = 0_i64, 2_i64, 1_i64
        do j = 0_i64, 3_i64, 1_i64
          a(j, i) = i * 4_i64 + j
        end do
      end do
      allocate(b(0:2_i64, 0:3_i64))
      allocate(c(0:2_i64, 0:3_i64))
      do i_0001 = 0_i64, 3_i64, 1_i64
        b(:, i_0001) = a(i_0001, :)
        c(:, i_0001) = a(i_0001, :)
      end do
      if (allocated(a)) then
        deallocate(a)
      end if
      if (allocated(b)) then
        deallocate(b)
      end if
      if (allocated(c)) then
        deallocate(c)
      end if

    end program prog_prog_tmp
    ```

-   C equivalent:

    ```C
    #include "tmp.h"
    #include <stdlib.h>
    #include "ndarrays.h"
    #include <stdint.h>
    int main()
    {
        t_ndarray a = {.shape = NULL};
        int64_t i;
        int64_t j;
        t_ndarray b = {.shape = NULL};
        t_ndarray c = {.shape = NULL};
        int64_t i_0001;
        int64_t i_0002;
        a = array_create(2, (int64_t[]){3, 4}, nd_double);
        for (i = 0; i < 3; i += 1)
        {
            for (j = 0; j < 4; j += 1)
            {
                GET_ELEMENT(a, nd_double, (int64_t)i, (int64_t)j) = i * 4 + j;
            }
        }
        b = array_create(2, (int64_t[]){4, 3}, nd_double);
        c = array_create(2, (int64_t[]){4, 3}, nd_double);
        for (i_0001 = 0; i_0001 < 4; i_0001 += 1)
        {
            for (i_0002 = 0; i_0002 < 3; i_0002 += 1)
            {
                GET_ELEMENT(b, nd_double, (int64_t)i_0001, (int64_t)i_0002) = GET_ELEMENT(a, nd_double, (int64_t)i_0002, (int64_t)i_0001);
                GET_ELEMENT(c, nd_double, (int64_t)i_0001, (int64_t)i_0002) = GET_ELEMENT(a, nd_double, (int64_t)i_0002, (int64_t)i_0001);
            }
        }
        free_array(a);
        free_array(b);
        free_array(c);
        return 0;
    }
    ```

## Other functions

-   Supported [math functions](https://numpy.org/doc/stable/reference/routines.math.html) (optional parameters are not supported):

    sqrt, abs, sin, cos, exp, log, tan, arcsin, arccos, arctan, arctan2, sinh, cosh, tanh, arcsinh, arccosh and
    arctanh.

-   Supported [array creation routines](https://numpy.org/doc/stable/reference/routines.array-creation.html) (fully supported):

    -   empty, full, ones, zeros, arange (`like` parameter is not supported).
    -   empty_like, full_like, and zeros_like, ones_like (`subok` parameter is not supported).
    -   rand, randint
    -   where, count_nonzero (fortran only)
    -   nonzero (fortran only, 1D only)

-   others:

    -   amax, amin, sum, shape, size, floor, sign

If discrepancies beyond round-off error are found between [Numpy](https://numpy.org/doc/stable/reference/)'s and [Pyccel](https://github.com/pyccel/pyccel)'s results, please create an issue at <https://github.com/pyccel/pyccel/issues> and provide a small example of your problem. Do not forget to specify your target language.

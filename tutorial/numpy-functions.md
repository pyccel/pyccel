# Supported Numpy function by Pyccel

In Pyccel we try to support the most used Numpy functions by developers. here is some of them:

## [Norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

- Supported parameters:

    x: array_like
       Input array. If axis is None, x must be 1-D or 2-D, unless ord is None.
       If both axis and ord are None, the 2-norm of x.ravel will be returned.

    axis: {None, int, 2-tuple of ints}, optional.
         If axis is an integer, it specifies the axis of x along which to compute the vector norms.
         If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of
         these matrices are computed. If axis is None then either a vector norm (when x is 1-D) or a
         matrix norm (when x is 2-D) is returned. The default is None. New in version 1.8.0.

- Supported languages: Fortran

- python code:

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

- fortran equivalent:

    ```fortran
    program prog_test_norm

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(C_INT64_T), allocatable :: arr1(:)
    real(C_DOUBLE) :: nrm
    integer(C_INT64_T), allocatable :: arr2(:,:)
    real(C_DOUBLE), allocatable :: nrm2(:)

    allocate(arr1(0:4_C_INT64_T - 1_C_INT64_T))
    arr1 = [1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T]
    nrm = Norm2(Real(arr1, C_DOUBLE))
    print *, nrm
    allocate(arr2(0:4_C_INT64_T - 1_C_INT64_T, 0:2_C_INT64_T - 1_C_INT64_T))
    arr2 = reshape([[1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T], [ &
          4_C_INT64_T, 3_C_INT64_T, 2_C_INT64_T, 1_C_INT64_T]], [ &
          4_C_INT64_T, 2_C_INT64_T])
    allocate(nrm2(0:2_C_INT64_T - 1_C_INT64_T))
    nrm2 = Norm2(Real(arr2, C_DOUBLE),2_C_INT64_T - 1_C_INT64_T)
    print *, nrm

    end program prog_test_norm
    ```

## [Real](https://numpy.org/doc/stable/reference/generated/numpy.real.html) and [imag](https://numpy.org/doc/stable/reference/generated/numpy.imag.html) functions

- Supported languages: C (scalars only), fortran

- python code:

    ```python
    from numpy import imag, real, array
    arr1 = array([1+1j,2+1j,3+1j,4+1j])
    real_part = real(arr1)
    imag_part = imag(arr1)
    print("real part for arr1: " , real_part, "\nimag part for arr1: ", imag_part)
    ```

- fortran equivalent:

    ```fortran
    program prog_test_imag_real

    use, intrinsic :: ISO_C_BINDING

    implicit none

    complex(C_DOUBLE_COMPLEX) :: n
    real(C_DOUBLE) :: nreal_part
    real(C_DOUBLE) :: nimag_part

    n = (3.0_C_DOUBLE, 7.0_C_DOUBLE)
    nreal_part = Real(n, C_DOUBLE)
    nimag_part = aimag(n)
    print *, 'real part for n : ' // ' ' , nreal_part, ACHAR(10) // 'imag part for n: ' // ' ' , nimag_part

    end program prog_test_imag_real
    ```

- C equivalent:

    ```C
    #include <complex.h>
    #include <stdlib.h>
    #include <stdio.h>
    int main()
    {
        double complex n;
        double nreal_part;
        double nimag_part;

        n = (3.0 + 7.0 * _Complex_I);
        nreal_part = creal(n);
        nimag_part = cimag(n);
        printf("%s %.12lf %s %.12lf\n", "real part for n : ", nreal_part, "\nimag part for n: ", nimag_part);
        return 0;
    }
    ```

- python code with arrays:

    ```python
    from numpy import imag, real, array
    arr1 = array([1+1j,2+1j,3+1j,4+1j])
    real_part = real(arr1)
    imag_part = imag(arr1)
    print("real part for arr1: " , real_part, "\nimag part for arr1: ", imag_part)
    ```

- fortran equivalent:

    ```fortran
    program prog_test_imag_real

    use, intrinsic :: ISO_C_BINDING

    implicit none

    complex(C_DOUBLE_COMPLEX), allocatable :: arr1(:)
    real(C_DOUBLE), allocatable :: real_part(:)
    real(C_DOUBLE), allocatable :: imag_part(:)

    allocate(arr1(0:4_C_INT64_T - 1_C_INT64_T))
    arr1 = [(1.0_C_DOUBLE, 1.0_C_DOUBLE), (2.0_C_DOUBLE, 1.0_C_DOUBLE), ( &
          3.0_C_DOUBLE, 1.0_C_DOUBLE), (4.0_C_DOUBLE, 1.0_C_DOUBLE)]
    allocate(real_part(0:4_C_INT64_T - 1_C_INT64_T))
    real_part = Real(arr1, C_DOUBLE)
    allocate(imag_part(0:4_C_INT64_T - 1_C_INT64_T))
    imag_part = aimag(arr1)
    print *, 'real part for arr1: ' // ' ' , real_part, ACHAR(10) // 'imag part for arr1: ' // ' ' , imag_part

    end program prog_test_imag_real
    ```

## [Prod](https://numpy.org/doc/stable/reference/generated/numpy.prod.html)

- Supported parameters:

    a: array_like,
        Input data.

- Supported languages: fortran

- python code:

    ```python
    from numpy import array, prod

    arr = array([1,2,3,4])
    prd = prod(arr)
    print("prd: ", prd)
    ```

- fortran equivalent:

    ```fortran
    program prog_test_prod

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(C_INT64_T), allocatable :: arr(:)
    integer(C_INT64_T) :: prd

    allocate(arr(0:4_C_INT64_T - 1_C_INT64_T))
    arr = [1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T]
    prd = product(arr)
    print *, 'prd: ' // ' ' , prd

    end program prog_test_prod
    ```

## [mod](https://numpy.org/doc/stable/reference/generated/numpy.mod.html)

- Supported parameters:

    x1: array_like
        Dividend array.

    x2: array_like,
        Divisor array. If x1.shape != x2.shape, they must be
        broadcastable to a common shape (which becomes the shape of the output).

- Supported language: fortran.

- python code:

    ```python
    from numpy import array, mod

    arr = array([1,2,3,4])
    res = mod(arr, arr)
    print("res: ", res)
    ```

- fortran equivalent:

    ```fortran
    program prog_test_mod

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(C_INT64_T), allocatable :: arr(:)
    integer(C_INT64_T), allocatable :: res(:)

    allocate(arr(0:4_C_INT64_T - 1_C_INT64_T))
    arr = [1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T]
    allocate(res(0:4_C_INT64_T - 1_C_INT64_T))
    res = modulo(arr,arr)
    print *, 'res: ' // ' ' , res

    end program prog_test_mod
    ```

## [matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)

- Supported parameters:

    x1, x2: array_like,
    Input arrays, scalars not allowed.

- Supported langauges: fortran

- python code:

    ```python
    from numpy import array, matmul

    arr = array([[1,2],[3,4]])
    res = matmul(arr, arr)
    print("res: ", res)
    ```

- fortran equivalent:

    ```fortran
    program prog_test_matmul

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(C_INT64_T), allocatable :: arr(:,:)
    integer(C_INT64_T), allocatable :: res(:,:)

    allocate(arr(0:2_C_INT64_T - 1_C_INT64_T, 0:2_C_INT64_T - 1_C_INT64_T))
    arr = reshape([[1_C_INT64_T, 2_C_INT64_T], [3_C_INT64_T, 4_C_INT64_T]], &
          [2_C_INT64_T, 2_C_INT64_T])
    allocate(res(0:2_C_INT64_T - 1_C_INT64_T, 0:2_C_INT64_T - 1_C_INT64_T))
    res = matmul(arr,arr)
    print *, 'res: ' // ' ' , res

    end program prog_test_matmul
    ```

## [Numpy types](https://numpy.org/devdocs/user/basics.types.html)

- Supported types : bool, int, int8, int16, int32, int64, float, float32, float64, complex64 and complex128. they can be used as cast functions too.

## Other functions

- Supported [math functions](https://numpy.org/doc/stable/reference/routines.math.html):

    NumpySqrt, NumpyAbs, NumpySin, NumpyCos, NumpyExp, NumpyLog, NumpyTan, NumpyArcsin, NumpyArccos, NumpyArctan, NumpyArctan2, NumpySinh, NumpyCosh, NumpyTanh, NumpyArcsinh, NumpyArccosh and
    NumpyArctanh.

- Supported arrays manipulation:

    NumpyEmpty, NumpyEmptyLike, NumpyFull, NumpyFullLike, NumpyOnes,
    NumpyOnesLike, NumpyZeros and NumpyZerosLike.

- others:

    NumpyMax, NumpyMin, NumpySum, NumpyRand, NumpyRandint, Shape, NumpyArange, NumpyFloor.

You can get more informations about the behaviour of each function in [Numpy](https://numpy.org/doc/stable/reference/) documentation.
Please if you face different behaviour between Numpy results and Pyccel results, create an issue at https://github.com/pyccel/pyccel/issues and provide a small example of your problem. Do not forget to specify your target language.

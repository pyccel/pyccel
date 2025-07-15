# Supported NumPy function by Pyccel

In Pyccel we try to support the NumPy functions which developers use the most.. Here are some of them:

## [norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

-   Supported parameters:

    ```python
    x: array_like
       Input array. If axis is None, x must be 1-D or 2-D, unless ord is None.
       If both axis and ord are None, the 2-norm of x.ravel will be returned.

    axis: {None, int, 2-tuple of ints}, optional.
         If axis is an integer, it specifies the axis of x along which to compute the vector norms.
         If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of
         these matrices are computed. If axis is None then either a vector norm (when x is 1-D) or a
         matrix norm (when x is 2-D) is returned. The default is None. New in version 1.8.0.
    ```

-   Supported languages: Fortran (2-norm)

-   Python code:

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

-   Fortran equivalent:

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

-   Supported languages: C, Fortran

-   Python code:

    ```python
    from numpy import imag, real, array

    if __name__ == '__main__':
        arr1 = array([1+1j,2+1j,3+1j,4+1j])
        real_part = real(arr1)
        imag_part = imag(arr1)
        print("real part for arr1: " , real_part, "\nimag part for arr1: ", imag_part)
    ```

-   Fortran equivalent:

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
    int main()
    {
        array_double_complex_1d arr1 = {0};
        array_double_1d real_part = {0};
        array_double_1d imag_part = {0};
        int64_t i;
        double complex* arr1_ptr;
        double* real_part_ptr;
        double* imag_part_ptr;
        int64_t i_0001;
        int64_t i_0002;
        arr1_ptr = malloc(sizeof(double complex) * (INT64_C(4)));
        arr1 = (array_double_complex_1d)cspan_md_layout(c_ROWMAJOR, arr1_ptr, INT64_C(4));
        (*cspan_at(&arr1, INT64_C(0))) = (1.0 + 1.0 * _Complex_I);
        (*cspan_at(&arr1, INT64_C(1))) = (2.0 + 1.0 * _Complex_I);
        (*cspan_at(&arr1, INT64_C(2))) = (3.0 + 1.0 * _Complex_I);
        (*cspan_at(&arr1, INT64_C(3))) = (4.0 + 1.0 * _Complex_I);
        real_part_ptr = malloc(sizeof(double) * (INT64_C(4)));
        real_part = (array_double_1d)cspan_md_layout(c_ROWMAJOR, real_part_ptr, INT64_C(4));
        for (i = INT64_C(0); i < INT64_C(4); i += INT64_C(1))
        {
            (*cspan_at(&real_part, i)) = creal((*cspan_at(&arr1, i)));
        }
        imag_part_ptr = malloc(sizeof(double) * (INT64_C(4)));
        imag_part = (array_double_1d)cspan_md_layout(c_ROWMAJOR, imag_part_ptr, INT64_C(4));
        for (i = INT64_C(0); i < INT64_C(4); i += INT64_C(1))
        {
            (*cspan_at(&imag_part, i)) = cimag((*cspan_at(&arr1, i)));
        }
        printf("real part for arr1:  ");
        printf("[");
        for (i_0001 = INT64_C(0); i_0001 < INT64_C(3); i_0001 += INT64_C(1))
        {
            printf("%.15lf ", (*cspan_at(&real_part, i_0001)));
        }
        printf("%.15lf]", (*cspan_at(&real_part, INT64_C(3))));
        printf("\nimag part for arr1:  ");
        printf("[");
        for (i_0002 = INT64_C(0); i_0002 < INT64_C(3); i_0002 += INT64_C(1))
        {
            printf("%.15lf ", (*cspan_at(&imag_part, i_0002)));
        }
        printf("%.15lf]\n", (*cspan_at(&imag_part, INT64_C(3))));
        free(arr1.data);
        arr1.data = NULL;
        free(real_part.data);
        real_part.data = NULL;
        free(imag_part.data);
        imag_part.data = NULL;
        return 0;
    }
    ```

## [prod](https://numpy.org/doc/stable/reference/generated/numpy.prod.html)

-   Supported parameters:

    ```python
    a: array_like,
        Input data.
    ```

-   Supported languages: Fortran

-   Python code:

    ```python
    from numpy import array, prod

    arr = array([1,2,3,4])
    prd = prod(arr)
    print("prd: ", prd)
    ```

-   Fortran equivalent:

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

    ```python
    x1: array_like
        Dividend array.

    x2: array_like,
        Divisor array. If x1.shape != x2.shape, they must be
        broadcastable to a common shape (which becomes the shape of the output).
    ```

-   Supported language: Fortran.

-   Python code:

    ```python
    from numpy import array, mod

    arr = array([1,2,3,4])
    res = mod(arr, arr)
    print("res: ", res)
    ```

-   Fortran equivalent:

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

    ```python
    x1, x2: array_like,
        Input arrays (must be 1d or 2d), scalars not allowed.
    ```

-   Supported languages: Fortran (1d or 2d arrays only).

-   Python code:

    ```python
    from numpy import array, matmul

    arr = array([[1,2],[3,4]])
    res = matmul(arr, arr)
    print("res: ", res)
    ```

-   Fortran equivalent:

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

-   Supported languages: C, Fortran

-   Supported parameters:

    ```python
    start, stop: array_like,

    num: int, optional (Default is 50)

    endpoint: bool, optional (Default is True)

    dtype: dtype, optional
    ```

-   Python code:

    ```python
    from numpy import linspace

    if __name__ == "__main__":
        x = linspace(0, 10, 20, endpoint=True, dtype='float64')
        print(x)
    ```

-   Fortran equivalent:

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
      if (allocated(x)) deallocate(x)

    end program prog_prog_test
    ```

-   C equivalent:

    ```C
    int main()
    {
        array_double_1d x = {0};
        int64_t i;
        double* x_ptr;
        int64_t i_0001;
        x_ptr = malloc(sizeof(double) * (INT64_C(20)));
        x = (array_double_1d)cspan_md_layout(c_ROWMAJOR, x_ptr, INT64_C(20));
        for (i = INT64_C(0); i < INT64_C(20); i += INT64_C(1))
        {
            (*cspan_at(&x, i)) = (INT64_C(0) + i*(double)((INT64_C(10) - INT64_C(0))) / 19.0);
            (*cspan_at(&x, INT64_C(19))) = (double)INT64_C(10);
        }
        printf("[");
        for (i_0001 = INT64_C(0); i_0001 < INT64_C(19); i_0001 += INT64_C(1))
        {
            printf("%.15lf ", (*cspan_at(&x, i_0001)));
        }
        printf("%.15lf]\n", (*cspan_at(&x, INT64_C(19))));
        free(x.data);
        x.data = NULL;
        return 0;
    }
    ```

## [Transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)

-   Supported languages: C, Fortran

-   Supported parameters:

    ```python
    a: array_like,
    ```

-   Python code:

    ```python
    from numpy import transpose
    
    def print_transpose(y : 'int[:,:,:]'):
        z = transpose(y)
        b = y.T
        print(y[0,1,2], z[0,1,2], b[0,1,2])
    ```

-   Fortran equivalent:

    ```fortran
    !........................................
    subroutine print_transpose(y)
  
      implicit none
  
      integer(i64), target, intent(in) :: y(0_i64:, 0_i64:, 0_i64:)
      integer(i64), pointer :: z(:, :, :)
      integer(i64), pointer :: b(:, :, :)
  
      z(0:, 0:, 0:) => y
      b(0:, 0:, 0:) => y
      write(stdout, '(I0, A, I0, A, I0)', advance="yes") y(2_i64, 1_i64, &
            0_i64) , ' ' , z(0_i64, 1_i64, 2_i64) , ' ' , b(0_i64, 1_i64, &
            2_i64)
  
    end subroutine print_transpose
    !........................................
    ```

-   C equivalent:

    ```C
    /*........................................*/
    void print_transpose(array_int64_3d y)
    {
        array_int64_3d z = {0};
        array_int64_3d b = {0};
        z = cspan_slice(array_int64_3d, &y, {c_ALL}, {c_ALL}, {c_ALL});
        cspan_transpose(&z);
        b = cspan_slice(array_int64_3d, &y, {c_ALL}, {c_ALL}, {c_ALL});
        cspan_transpose(&b);
        printf("%"PRId64" %"PRId64" %"PRId64"\n", (*cspan_at(&y, INT64_C(0), INT64_C(1), INT64_C(2))), (*cspan_at(&z, INT64_C(0), INT64_C(1), INT64_C(2))), (*cspan_at(&b, INT64_C(0), INT64_C(1), INT64_C(2))));
    }
    /*........................................*/
    ```

## Other functions

-   Supported [math functions](https://numpy.org/doc/stable/reference/routines.math.html) (optional parameters are not supported):

    `sqrt`, `abs`, `sin`, `cos`, `exp`, `expm1`, `log`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`, `sinh`, `cosh`, `tanh`,
    `arcsinh`, `arccosh`, `arctanh`, `true_divide`, `divide`.

-   Supported [logic functions](https://numpy.org/doc/stable/reference/routines.logic.html) (optional parameters are not supported):

    `isfinite`, `isinf`, `isnan`

-   Supported [array creation routines](https://numpy.org/doc/stable/reference/routines.array-creation.html) (fully supported):

    -   `empty`, `full`, `ones`, `zeros`, `array`, `arange` (`like` parameter is not supported).
    -   `empty_like`, `full_like`, `zeros_like`, and `ones_like` (`subok` parameter is not supported).
    -   `array` (`copy`, `subok`, and `like` parameters are not supported).
    -   `rand`, `randint`
    -   `where`, `count_nonzero` (Fortran only)
    -   `nonzero` (Fortran only, 1D only)
    -   `copy` (`subok` parameter is not supported)

-   others:

    -   `amax`, `amin`, `sum`, `shape`, `size`, `floor`, `sign`, `result_type`

If discrepancies beyond round-off error are found between [NumPy](https://numpy.org/doc/stable/reference/)'s and [Pyccel](https://github.com/pyccel/pyccel)'s results, please create an issue at <https://github.com/pyccel/pyccel/issues> and provide a small example of your problem. Do not forget to specify your target language.

#include <vector/header.inc>
#define __IDENTITY(x) x

public :: __IDENTITY(Vector)_min
public :: __IDENTITY(Vector)_max

contains

#define __guard __vector_guard
#include "parameters/T/copy_vector_T_to_internal_T.inc"
#include "parameters/T/define_derived_macros.inc"

  subroutine __IDENTITY(Vector)_reverse(my_vector)
    class(Vector), intent(inout) :: my_vector

    integer :: left, right
    __T_declare_dummy__ :: temp

    left = 1
    right = my_vector%size()

    do while (left < right)
        temp = my_vector%at(left)
        call my_vector%set(left, my_vector%at(right))
        call my_vector%set(right, temp)

        left = left + 1
        right = right - 1
    end do

  end subroutine __IDENTITY(Vector)_reverse

  function __IDENTITY(Vector)_min(my_vector) result(result)
    class(Vector), intent(in) :: my_vector
    __T_declare_dummy__ :: result

    type(VectorIterator) :: iter
    type(VectorIterator) :: last

    iter = my_vector%begin()
    last = my_vector%end()

    if (iter == last) then
       print *, "Error: Empty vector, cannot determine min."
       stop
    end if

    result = iter%of()
    do while (iter /= last)
      if (T_LT(iter%of(), result)) then
        result = iter%of()
      end if
      call iter%next()
    end do

  end function __IDENTITY(Vector)_min

  function __IDENTITY(Vector)_max(my_vector) result(result)
    class(Vector), intent(in) :: my_vector
    __T_declare_dummy__ :: result

    type(VectorIterator) :: iter
    type(VectorIterator) :: last

    iter = my_vector%begin()
    last = my_vector%end()

    if (iter == last) then
       print *, "Error: Empty vector, cannot determine max."
       stop
    end if

    result = iter%of()
    do while (iter /= last)
      if (T_LT(result, iter%of())) then
        result = iter%of()
      end if
      call iter%next()
    end do

  end function __IDENTITY(Vector)_max

  subroutine __IDENTITY(Vector)_insert(my_vector, idx, value)
    use, intrinsic :: ISO_C_BINDING, only: i64 => C_INT64_T

    class(Vector), intent(inout)              :: my_vector
    integer(i64), intent(in)                  :: idx
    __T_declare_dummy__, intent(in)           :: value
  
    integer(i64) :: n, pos, i
    type(VectorIterator) :: iter
  
    n = my_vector%size()
    pos = idx
    iter = my_vector % begin()
  
    if (pos < 0) pos = pos + n
    if (pos < 0) pos = 0
    if (pos > n) pos = n
  
    iter =  my_vector % insert(iter + pos, value)
  end subroutine __IDENTITY(Vector)_insert

#include <vector/tail.inc>

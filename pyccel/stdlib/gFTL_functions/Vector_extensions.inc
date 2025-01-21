#include <vector/header.inc>

contains

#define __IDENTITY(x) x
#define __guard __vector_guard
#include "parameters/T/copy_vector_T_to_internal_T.inc"
#include "parameters/T/define_derived_macros.inc"

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
      if (iter%of() < result) then
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
      if (iter%of() > result) then
        result = iter%of()
      end if
      call iter%next()
    end do

  end function __IDENTITY(Vector)_max

#include <vector/tail.inc>

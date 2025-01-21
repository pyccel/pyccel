#include <vector/header.inc>

contains

#define __IDENTITY(x) x
#define __guard __vector_guard
#include "parameters/T/copy_vector_T_to_internal_T.inc"
#include "parameters/T/define_derived_macros.inc"


  subroutine __IDENTITY(Vector)_reverse(my_vector)
    class(Vector), intent(inout) :: my_vector

    type(VectorIterator) :: iter_start
    type(VectorIterator) :: iter_end
    __T_declare_dummy__ :: temp

    iter_start = my_vector%begin()
    iter_end = my_vector%end()
    call iter_end%prev()  ! Move end iterator to the last valid element

    do while (iter_start /= iter_end .and. iter_start%next() /= iter_end)
      temp = iter_start%of()
      call iter_start%set(iter_end%of())
      call iter_end%set(temp)

      call iter_start%next()
      call iter_end%prev()
    end do

  end subroutine __IDENTITY(Vector)_reverse

#ifdef __T_IS_COMPLEX
  #error "Complex types are not supported for min/max operations"
#else

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
#endif
#include <vector/tail.inc>

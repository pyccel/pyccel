#include <set/header.inc>

contains

#define __IDENTITY(x) x
#define __guard __set_guard
#include "parameters/T/copy_set_T_to_internal_T.inc"
#include "parameters/T/define_derived_macros.inc"

  function __IDENTITY(Set)_pop(my_set) result(result)
    class(Set), intent(inout) :: my_set
    __T_declare_dummy__ :: result
  
    type(SetIterator) :: iter1
    type(SetIterator) :: iter2
  
    iter1 = my_set%begin()
  
    result = iter1%of()
  
    iter2 = my_set%erase(iter1)
  
  end function __IDENTITY(Set)_pop

  subroutine __IDENTITY(Set)_intersection_update(this, other_set)
    class(Set), intent(inout) :: this
    class(Set), intent(in) :: other_set

    type(SetIterator) :: iter
    type(SetIterator) :: last

    iter = this%begin()
    last = this%end()
    do while (iter /= last)
      if (other_set % count(iter%of()) == 0) then
        iter = this%erase(iter)
      else
        call iter%next()
      end if
    end do

  end subroutine __IDENTITY(Set)_intersection_update

   function __IDENTITY(Set)_min(my_set) result(result)
    class(Set), intent(in) :: my_set
    __T_declare_dummy__ :: result

    type(SetIterator) :: iter
    type(SetIterator) :: last

    iter = my_set%begin()
    last = my_set%end()

    result = iter%of()
  end function __IDENTITY(Set)_min

  function __IDENTITY(Set)_max(my_set) result(result)
    class(Set), intent(in) :: my_set
    __T_declare_dummy__ :: result

    type(SetIterator) :: iter
    type(SetIterator) :: last

    iter = my_set%begin()
    last = my_set%end()


    call last%prev()

    result = last%of()
  end function __IDENTITY(Set)_max

#include <set/tail.inc>

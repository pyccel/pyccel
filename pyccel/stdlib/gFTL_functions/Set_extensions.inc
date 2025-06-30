
#include <set/header.inc>
#define __IDENTITY(x) x

public :: __IDENTITY(Set)_pop
public :: __IDENTITY(Set)_intersection_update
public :: __IDENTITY(Set)_is_disjoint
public :: __IDENTITY(Set)_min
public :: __IDENTITY(Set)_max

contains

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

  subroutine __IDENTITY(Set)_difference_update(this, other_set)
    class(Set), intent(inout) :: this
    class(Set), intent(in) :: other_set

    type(SetIterator) :: iter
    type(SetIterator) :: last

    iter = this%begin()
    last = this%end()
    do while (iter /= last)
      if (other_set % count(iter%of()) == 0) then
        call iter%next()
      else
        iter = this%erase(iter)
      end if
    end do

  end subroutine __IDENTITY(Set)_difference_update

  function __IDENTITY(Set)_is_disjoint(this, other_set) result(Out)
    use, intrinsic :: ISO_C_Binding, only : b1 => C_BOOL
    class(Set), target, intent(in) :: this
    class(Set), target, intent(in) :: other_set
    logical(b1) :: Out

    class(Set), pointer :: shortest
    class(Set), pointer :: longest
    type(SetIterator) :: iter
    type(SetIterator) :: last

    if (this % size() > other_set % size()) then
      shortest => other_set
      longest => this
    else
      shortest => this
      longest => other_set
    endif

    iter = shortest%begin()
    last = shortest%end()
    do while (iter /= last)
      if (longest % count(iter%of()) /= 0) then
        Out = .false.
        return
      else
        call iter%next()
      end if
    end do

    Out = .true.

  end function __IDENTITY(Set)_is_disjoint

  ! Sets in gFTL are BSTs, therefore the min is the first element
   function __IDENTITY(Set)_min(my_set) result(result)
    class(Set), intent(in) :: my_set
    __T_declare_dummy__ :: result

    type(SetIterator) :: iter

    iter = my_set%begin()

    result = iter%of()
  end function __IDENTITY(Set)_min

  ! Sets in gFTL are BSTs, therefore the max is the last element
  function __IDENTITY(Set)_max(my_set) result(result)
    class(Set), intent(in) :: my_set
    __T_declare_dummy__ :: result

    type(SetIterator) :: last

    last = my_set%end()

    call last%prev()

    result = last%of()
  end function __IDENTITY(Set)_max

#include <set/tail.inc>

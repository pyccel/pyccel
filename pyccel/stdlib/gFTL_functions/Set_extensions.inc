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

#include <set/tail.inc>

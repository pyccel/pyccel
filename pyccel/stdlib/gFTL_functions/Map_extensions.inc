#include <map/header.inc>

contains

#define __IDENTITY(x) x
#define __guard __map_guard
#include "parameters/Key/copy_map_Key_to_internal_Key.inc"
#include "parameters/T/copy_map_T_to_internal_T.inc"
#include "parameters/Key/define_derived_macros.inc"
#include "parameters/T/define_derived_macros.inc"

  function __IDENTITY(Map)_pop(my_map, key) result(value)
    class(Map), intent(inout) :: my_map
    __Key_declare_dummy__, intent(in) :: key

    __T_declare_dummy__ :: value

    type (__map_iterator) :: iter

    iter = my_map % find( key )
    value = iter % second()
    iter = my_map % erase( iter )

  end function __IDENTITY(Map)_pop

  function __IDENTITY(Map)_pop_with_default(my_map, key, default) result(value)
    class(Map), intent(inout) :: my_map
    __Key_declare_dummy__, intent(in) :: key
    __T_declare_dummy__, intent(in) :: default

    __T_declare_dummy__ :: value
    type (__map_iterator) :: iter

    iter = my_map % find( key )

    if (iter /= my_map % end()) then
      value = iter % second()
      iter = my_map % erase( iter )
    else
      value = default
    endif

  end function __IDENTITY(Map)_pop_with_default

  subroutine __IDENTITY(Map)_popitem(my_map, key, value)
    class(Map), intent(inout) :: my_map
    __Key_declare_dummy__, intent(out) :: key
    __T_declare_dummy__, intent(out) :: value

    type (__map_iterator) :: iter

    iter =  my_map % begin()

    key = iter % first()
    value = iter % second()

    iter = my_map % erase( iter )

  end subroutine __IDENTITY(Map)_popitem

#include <map/tail.inc>


module class_renaming
  use, intrinsic :: iso_c_binding, only : i64 => C_INT64_T

  implicit none

  type :: CounterWithLongName
    integer(i64) :: value
  contains
    procedure :: create => counter_create
    procedure :: free => counter_free
    procedure :: increment => counter_increment
    procedure :: get_value => counter_get_value
  end type CounterWithLongName

contains

  subroutine counter_create(this, start)
    class(CounterWithLongName), intent(inout) :: this
    integer(i64), intent(in) :: start
    this%value = start
  end subroutine counter_create

  subroutine counter_free(this)
    class(CounterWithLongName), intent(inout) :: this
    this%value = -1
  end subroutine counter_free

  subroutine counter_increment(this)
    class(CounterWithLongName), intent(inout) :: this
    this%value = this%value + 1
  end subroutine counter_increment

  function counter_get_value(this) result(v)
    class(CounterWithLongName), intent(in) :: this
    integer(i64) :: v
    v = this%value
  end function counter_get_value

end module class_renaming


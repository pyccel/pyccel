module array_methods
  use, intrinsic :: iso_c_binding
  implicit none

  type :: ArrayOps
    real(c_double), allocatable :: data(:)
  contains
    procedure :: create => create_arrayops
    procedure :: free => free_arrayops
    procedure :: set_data => array_set_data
    procedure :: sum => array_sum
    procedure :: scale => array_scale
  end type ArrayOps

contains

  subroutine create_arrayops(obj)
    class(ArrayOps), intent(inout) :: obj
  end subroutine create_arrayops

  subroutine free_arrayops(obj)
    class(ArrayOps), intent(inout) :: obj
    if (allocated(obj%data)) then
      deallocate(obj%data)
    end if
  end subroutine free_arrayops

  subroutine array_set_data(this, arr, n)
    class(ArrayOps), intent(inout) :: this
    real(c_double), intent(in) :: arr(n)
    integer(c_int), intent(in) :: n
    if (allocated(this%data)) deallocate(this%data)
    allocate(this%data(n))
    this%data = arr
  end subroutine array_set_data

  function array_sum(this) result(total)
    class(ArrayOps), intent(in) :: this
    real(c_double) :: total
    integer :: i
    total = 0.0d0
    do i = 1, size(this%data)
      total = total + this%data(i)
    end do
  end function array_sum

  subroutine array_scale(this, factor)
    class(ArrayOps), intent(inout) :: this
    real(c_double), intent(in) :: factor
    integer :: i
    do i = 1, size(this%data)
      this%data(i) = this%data(i) * factor
    end do
  end subroutine array_scale

end module array_methods

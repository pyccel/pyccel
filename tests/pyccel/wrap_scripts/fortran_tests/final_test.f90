module final_test
  type :: t
     real a,b
     real, pointer :: c(:), d(:)
   contains
     procedure :: init => t_init
     final :: t_finalizer
  end type

contains
  subroutine t_init(x)
    class(t) :: x
    allocate(x%c(1:5), source=[1.0, 5.0, 10.0, 15.0, 20.0])
    nullify(x%d)
  end subroutine t_init

  subroutine t_finalizer(x)
    type(t) :: x

    print *, 'entering t_finalizer'
    if (associated(x%c)) then
       print *, ' c allocated, cleaning up'
       deallocate(x%c)
    end if
    if (associated(x%d)) then
       print *, ' d allocated, cleaning up'
       deallocate(x%d)
    end if
  end subroutine
end module

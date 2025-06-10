"""
Pyccel header for OpenACC.
 OpenACC directives and Constructs are handled by the parser (see openacc.tx) and are parts of the Pyccel language.
 We only list here what can not be described in the openacc grammar.

 TODO - devicetype is for the moment an integer. => need to add precision
      - change * to h_void* and d_void* in 
         + acc_map_data
         + acc_unmap_data  
         + acc_deviceptr
         + acc_hostptr
         + acc_memcpy_to_device
         + acc_memcpy_to_device_async
         + acc_memcpy_from_device
         + acc_memcpy_from_device_async
         + acc_memcpy_device
         + acc_memcpy_device_async
         + 
      - data movement routines (do we really need them?)
from typing import Any
"""
#$ header metavar module_name='openacc'
#$ header metavar module_version='2.5'
#$ header metavar save=True
#$ header metavar external=False

# ............................................................
#            Runtime Library Routines for Fortran
# ............................................................
def acc_get_num_devices(anon_0 : 'int') -> 'int':
    ...

def acc_set_device_type(anon_0 : 'int') -> None:
    ...

def acc_get_device_type() -> 'int':
    ...

def acc_set_device_num(anon_0 : 'int', anon_1 : 'int') -> None:
    ...

def acc_get_device_num(anon_0 : 'int') -> 'int':
    ...

def acc_init(anon_0 : 'int') -> None:
    ...

def acc_shutdown(anon_0 : 'int') -> None:
    ...

def acc_async_test(anon_0 : 'bool') -> 'bool':
    ...

def acc_async_test_all() -> 'bool':
    ...

def acc_wait(anon_0 : 'bool') -> None:
    ...

def acc_wait_all(anon_0 : 'bool') -> None:
    ...

def acc_wait_async(anon_0 : 'bool', anon_1 : 'bool') -> None:
    ...

def acc_wait_all_async() -> None:
    ...

def acc_get_default_async() -> 'int':
    ...

def acc_set_default_async() -> None:
    ...

def acc_on_device(anon_0 : 'int') -> None:
    ...

def acc_malloc(anon_0 : 'int') -> 'int':
    ...

def acc_free(anon_0 : 'int') -> None:
    ...

def acc_map_data(anon_0 : 'Any', anon_1 : 'Any', anon_2 : 'int') -> None:
    ...

def acc_unmap_data(anon_0 : 'Any') -> None:
    ...

def acc_deviceptr(anon_0 : 'Any') -> 'int':
    ...

def acc_hostptr(anon_0 : 'Any') -> 'int':
    ...

def acc_memcpy_to_device(anon_0 : 'Any', anon_1 : 'Any', anon_2 : 'int') -> None:
    ...

def acc_memcpy_to_device_async(anon_0 : 'Any', anon_1 : 'Any', anon_2 : 'int', anon_3 : 'int') -> None:
    ...

def acc_memcpy_from_device(anon_0 : 'Any', anon_1 : 'Any', anon_2 : 'int') -> None:
    ...

def acc_memcpy_from_device_async(anon_0 : 'Any', anon_1 : 'Any', anon_2 : 'int', anon_3 : 'int') -> None:
    ...

def acc_memcpy_device(anon_0 : 'Any', anon_1 : 'Any', anon_2 : 'int') -> None:
    ...

def acc_memcpy_device_async(anon_0 : 'Any', anon_1 : 'Any', anon_2 : 'int', anon_3 : 'int') -> None:
    ...


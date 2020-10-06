# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

from pyccel.stdlib.internal.openacc import acc_get_device_type
from pyccel.stdlib.internal.openacc import acc_get_num_devices

dev_kind = acc_get_device_type()
dev_num  = acc_get_num_devices(dev_kind)

print(' number of available OpenACC devices :', dev_num)
print(' type of available OpenACC devices   :', dev_kind)


# pylint: disable=missing-function-docstring, missing-module-docstring, unused-variable
def create_list():
    a = [1,2,3]

def create_set():
    a = {1,2,3}

def create_dict():
    a = {1:1,2:2,3:3}

def list_reassign():
    a = [1,2,3]
    a = [1,2,3,4]
    a = [2 * i for i in range(15)]
    a = [1,2]

def set_reassign():
    a = {1,2,3}
    a = {1,2}
    a = {1,2,3,4}
    a = {1,2,3,4,5}

def dict_reassign():
    a = {1:1,2:2,3:3}
    a = {1:1,2:2,3:3,4:4}
    a = {1:1,2:2}
    a = {1:1,2:2,3:3,4:4,5:5}

def str_reassign():
    a = 'hello'
    a = 'hello world'
    a = 'hello cruel world'

def conditional_list(b1: bool):
    if (b1):
        a = [1,2,3]
    a = [1,2,3,4]

def conditional_set(b1: bool):
    if (b1):
        a = {1,2,3}
    a = {1,2,3,4}

def conditional_dict(b1: bool):
    if (b1):
        a = {1:1,2:2,3:3}
    a = {1:1,2:2,3:3,4:4,5:5}

def slice_assign():
    a = [1,2,3,4]
    b = a[1:-1]

def list_return():
    a = [1,2,3]
    return a

if __name__ == '__main__':
    create_list()
    create_set()
    create_dict()
    list_reassign()
    set_reassign()
    dict_reassign()
    conditional_list(True)
    conditional_set(True)
    conditional_dict(True)
    conditional_list(False)
    conditional_set(False)
    conditional_dict(False)
    slice_assign()

    tmp = list_return()
    tmp = list_return()

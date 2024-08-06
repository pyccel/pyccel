# pylint: disable=missing-function-docstring, missing-module-docstring

def allocate_list_of_list():
    a : 'list[list[int]]' = [[1, 2], [3,4]]
    print(a[0][0])

def allocate_list_of_pointers():
    a = [1,2,3]
    b = [a]
    print(b[0][0])

def get_list_element_1d():
    a = [1,2,3,4]
    return a[2]

if __name__ == '__main__':
    allocate_list_of_list()
    allocate_list_of_pointers()
    elem = get_list_element_1d()

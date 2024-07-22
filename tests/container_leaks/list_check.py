# pylint: disable=missing-function-docstring, missing-module-docstring

def allocate_list_of_list():
    a : 'list[list[int]]' = [[1, 2], [3,4]]
    print(a[0][0])

if __name__ == '__main__':
    allocate_list_of_list()

# pylint: disable=missing-function-docstring, missing-module-docstring


def create_nested_list():
    a = [[1,2,3], [4,5,6]]
    print(a[0][0])

def create_nested_list2():
    a = [[[1,2],[3,4]], [[5,6],[7,8]]]
    print(a[0][0])

def create_nested_dict():
    a = {1:[1,2,3], 2:[4,5,6]}
    print(a[1][0])

def create_nested_dict2():
    a = {1:{True: [1,2], False: [3,4]}, 2:{True: [5,6], False: [7,8]}}
    print(a[2][False][1])

def nested_list_reassign():
    a = [[1,2,3], [4,5,6]]
    print(a[0][0])
    a = [[7,8], [9, 10], [11, 12]]
    print(a[0][0])

def nested_dict_reassign():
    a = {1:[1,2,3], 2:[4,5,6]}
    print(a[1][0])
    a = {3:[1,2], 4:[4,5], 6:[7,8]}
    print(a[3][0])

def nested_list_ref():
    a = [1,2]
    b = [3,4]
    c = [a,b]
    print(c[0][0])

def nested_list_ref_pop():
    a = [1,2]
    b = [3,4]
    c = [a,b]
    d = c.pop()
    print(d[0])

def nested_list_pop_reassign_ref():
    a = [1,2]
    b = [3,4]
    c = [a,b]
    d = c.pop()
    print(d[0])
    g = d
    e = [d,b]
    g = e.pop()
    print(g)

def nested_list_pop_temp():
    a = [[1,2,3], [4,5,6]]
    b = a.pop()
    print(b[0])

def nested_list_ref_and_val_pop():
    a = [1,2]
    c = [a,[3,4]]
    d = c.pop()
    print(d[0])

def return_nested_list_element():
    a = [[1,2,3], [4,5,6]]
    b = a.pop()
    return b

def get_nested_list_element():
    a = [[1,2,3], [4,5,6]]
    b = a[0]
    return b[0]+b[1]+b[2]

if __name__ == '__main__':
    create_nested_list()
    create_nested_list2()
    create_nested_dict()
    create_nested_dict2()
    nested_list_reassign()
    nested_dict_reassign()
    nested_list_ref()
    nested_list_ref_pop()
    nested_list_pop_temp()
    nested_list_ref_and_val_pop()
    nested_list_pop_reassign_ref()

    x = return_nested_list_element()

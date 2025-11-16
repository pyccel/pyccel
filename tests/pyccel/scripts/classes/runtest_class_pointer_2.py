from class_pointer_2 import A, B

if __name__ == '__main__':
    a = A()
    b = B(a)

    a_2 = b.a

    print(a_2.x)
    print(a_2.y)


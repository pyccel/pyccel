from pyccel.decorators import kernel, types

@kernel
@types('int')
def func(a):
    print("Hello World! ", a)

if __name__ == '__main__':
    arr = 0
    func[1, 5](arr)

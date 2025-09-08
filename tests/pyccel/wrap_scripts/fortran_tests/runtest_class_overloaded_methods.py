from class_overloaded_methods import Adder

def main():
    a = Adder()
    print(a.add(3, 4))      # expected 7
    print(a.add(2.5, 3.5))  # expected 6.0

if __name__ == "__main__":
    main()

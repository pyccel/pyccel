# Development Conventions

There are a few conventions that are used in Pyccel and come up regularly during PR reviews. We try to document those conventions here and explain the reasoning behind them.

## Define variables for class properties

Accessing the property of a class has a small associated cost. As Python is an interpreted language, these costs add up. Therefore if you are going to use the same property multiple times in a section of code it is usually better (and often more readable) to define it once before beginning.

E.g:
```python
prec = self.precision
dtype = self.dtype
if prec in (None, -1):
    return LiteralString(f"<class '{dtype}'>")

precision = prec * (16 if dtype is NativeComplex() else 8)
if self._obj.rank > 0:
    return LiteralString(f"<class 'numpy.ndarray' ({dtype}{precision})>")
else:
    return LiteralString(f"<class 'numpy.{dtype}{precision}'>")
```
should be preferred over:
```python
dtype = str(self.dtype)
if self.precision in (None, -1):
    return LiteralString(f"<class '{dtype}'>")

precision = self.precision * (16 if self.dtype is NativeComplex() else 8)
if self._obj.rank > 0:
    return LiteralString(f"<class 'numpy.ndarray' ({dtype}{precision})>")
else:
    return LiteralString(f"<class 'numpy.{dtype}{precision}'>")
```

## Disabling Pylint

During the PR review Pylint is used to ensure that best coding practices are followed. It can occasionally be tempting to disable these errors rather than fixing the problem. However the Pylint preferences are already curated to ignore errors that we feel are unhelpful (e.g. variable name too long). This should therefore be avoided whenever possible. We allow Pylint errors to be disabled for the following 2 reasons:

### Disabling docstring checks in the tests

We do not enforce docstrings in the tests as it is hard to find a distinct way to describe each test and the pursuit of such distinct docstrings is time-consuming and ultimately does not add much value.

We therefore ask developers to add the following line to the top of every file in the `tests/` folder:

```python
# pylint: disable=missing-function-docstring, missing-module-docstring/
```

### Disabling false positives

Very occasionally Pylint will raise a false positive. In this case the errors can be ignored on a line by line basis, e.g:
```python
return self._attribute_nodes # pylint: disable=no-member
```

It is important to not disable errors file-wide as this will likely lead to bad practices being missed.

## Slots

In Pyccel we encourage the use of [slots](https://wiki.python.org/moin/UsingSlots) for small classes. In practice this means that their use is enforced in the `pyccel/ast` folder.

If you are not familiar with slots there is lots of documentation available online, e.g. [here](https://towardsdatascience.com/understand-slots-in-python-e3081ef5196d). Particular attention should be paid to how slots behave with inheritance.

## Class variables vs. Instance variables

Classes can contain both class variables and instance variables. E.g:
```python
class MyClass:
    __slots__ = ('myInstanceVar',)
    myClassVar = 0

    def __init__(self):
        self.myInstanceVar = 1
```

In order to save memory any variables which will remain constant over all instances of a class should be class variables. This discussion usually arises for the properties of the [`PyccelAstNode`](../pyccel/ast/basic.py) and its sub classes. The variables `_dtype`, `_precision`, `_rank`, `_shape`, and `_order` are often seen as instance variables, however for some classes (e.g. [`LiteralInteger`](../pyccel/ast/literals.py)) the values are known and never change. In this case they can be stored as class variables.

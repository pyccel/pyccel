# Headers and Decorators

## Template
A **template** in Pyccel, is used to allow the same function to take arguments of different types from a selection of types the user specifies.
#### The usage
In this example the argument **a**, could either be an integer or float, and the same for the argument **b**:
```python
from pyccel.decorators import template
@template('T', types=[int, float])
@template('Z', types=[int, float])
def f(a : 'T', b : 'Z'):
    pass
```
In this example The arguments **a** and **b**, should both be integers or floats at the same time:
```python
from pyccel.decorators import template
@template('T', types=[int, float])
def f(a : 'T', b : 'T'):
    pass
```
When  a function is decorated with the template decorator:
-   The templates are only available to the decorated function.
-   If the function is decorated with two templates with the same name, the first one gets overridden.
##### Examples
In this example the arguments of **f** can either be boolean or complex, they can not be integer or float.
```python
from pyccel.decorators import template
@template(name='T', types=['int','float'])
@template(name='T', types=['bool','complex'])
def f(a : 'T', b : 'T'):
    pass
```

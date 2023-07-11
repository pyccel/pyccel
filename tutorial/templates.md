# Headers and Decorators

## Template
### Templates using header comments
A **template** in Pyccel, is used to allow the same function to take arguments of different types from a selection of types the user specifies.
#### The usage
In this example the argument **a**, could either be an integer or float, and the same for the argument **b**:
```python
#$ header template T(int|real)
#$ header template Z(int|real)
#$ header function f(T, Z)
def f(a,b):
	pass
```
In this example The arguments **a** and **b**, should both be integers or floats at the same time:
```python
#$ header template T(int|real)
#$ header function f(T, T)
def f(a,b):
	pass
```
When a function is declared using a header comment as above:
-   The function can access the templates declared in its parent's scopes.
-   A template declared in the same scope as the function overrides any template with the same name in the parent's scopes.
#### Examples
In this example the template **T** can be used to define the arguments types of the nested function **f2**:
```python
#$ header template T(int|real)
def f1():
	#$ header f2(T, T)
	def f2(a, b):
		pass
	pass
```
In this example the arguments of **f2** can either be boolean or complex, they can not be integer or float:
```python
#$ header template T(int|real)
def f1():
	#$ header template T(bool|complex)
	#$ header f2(T, T)
	def f2(a, b):
		pass
	pass
```
### Templates using decorators
#### The usage
```python
from pyccel.decorators import types, template
@types('T', 'T')
@template(name='T', types=['int','real'])
def f(a,b):
	pass
```
Arguments:
-   name: the name of the template
-   types: the types the template represents.
---
*Note:*
The arguments **name** and **types** could also be passed of the form
`@template(T, types=['int', 'real'])` without specifying the keywords.and the order in which the decorators are called is not important.

---
When  a function is decorated with the template decorator:
-   The templates are only available to the decorated function.
-   The templates overrides any existing templates with the same name (declared as header comment).
-   If the function is decorated with two templates with the same name, the first one gets overridden.
##### Examples
In this example the arguments of **f** can either be boolean or complex, they can not be integer or float.
```python
from pyccel.decorators import types, template
#$ header template T(int|real)
@types('T', 'T')
@template(name='T', types=['bool','complex'])
def f(a,b):
  pass
```

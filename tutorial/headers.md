# Headers and Decorators

### Template
#### Templates using header comments:

A **template** in pyccel, is used to allow the same function to take arguments of different types from a selection of types the user specifies.

##### The usage:
```
#$ header template T(int|real)
#$ header template Z(int|real)
#$ header function f(T, Z)
def f(a,b):
	pass
```
In this example the argument **a**, could either be integer or float, and the same for the argument **b**.

```
#$ header template T(int|real)
#$ header function f(T, T)
def f(a,b):
	pass
```
---
*Note:*
In this example The arguments **a** and **b**, should both be integers or floats at the same time.

---
When a function is declared using a header comment as above:
- The function can access the templates declared in its parent's scopes.
- A template declared in the same scope as the function overrides any template with the same name in the parent's scopes.
##### Examples
```
#$ header template T(int|real)
def f1():
	#$ header f2(T, T)
	def f2(a, b):
		pass
	pass
```
The template **T** can be used to define the arguments types of the function **f2**
```
#$ header template T(int|real)
def f1():
	#$ header template T(bool|complex)
	#$ header f2(T, T)
	def f2(a, b):
		pass
	pass
```
The arguments of **f2** can either be bool or complex, they can not be int or float.

#### Templates using decorators:

##### The usage:
```
from pyccel.decorators import types, template
@types('T', 'T')
@template(name='T', types=['int','real'])
def f(a,b):
	pass
```
Arguments:
- name: the name of the template
- types: the types the tamplate represents.

---
*Note:*
The arguments **name** and **types** could also be passed of the form `@template(T, types=['int', 'real'])` without specifying the keywords.
The order in which the decorators are called is not important.

---

When  a function is decorated with the template decorator:
- The templates are only available to the decorated function.
- The templates overrides any existing templates with the same name (declared as header comment).
- If the function is decorated with two templates with the same name, the first one gets overrided.
##### Examples
```
from pyccel.decorators import types, template
#$ header template T(int|real)
@types('T', 'T')
@template(name='T', types=['bool','complex'])
def f(a,b):
	pass
```
The arguments of **f** can either be bool or complex, they can not be int or float.

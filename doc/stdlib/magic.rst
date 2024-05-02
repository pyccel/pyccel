
.. _magic-methods:



Magic methods
=============

More details can be found `here <http://www.diveintopython3.net/special-method-names.html>`_

====================  ========================  =========================  =========================   =================
..                    ..                           Magic methods           ..                          ..
====================  ========================  =========================  =========================   =================
:func:`__abs__`       :func:`__ge__`            :func:`__itruediv__`       :func:`__reversed__`        :func:`__trunc__`
:func:`__add__`       :func:`__get__`           :func:`__ixor__`           :func:`__rfloordiv__`       :func:`__xor__`
:func:`__and__`       :func:`__getattr__`       :func:`__instancecheck__`  :func:`__rlshift__`         .. 
:func:`__bool__`      :func:`__getattribute__`  :func:`__len__`            :func:`__rmod__`            ..
:func:`__bytes__`     :func:`__getitem__`       :func:`__lshift__`         :func:`__rmul__`            ..
:func:`__call__`      :func:`__getstate__`      :func:`__lt__`             :func:`__ror__`             ..
:func:`__ceil__`      :func:`__gt__`            :func:`__le__`             :func:`__round__`           ..
:func:`__complex__`   :func:`__hash__`          :func:`__mod__`            :func:`__rpow__`            ..
:func:`__contains__`  :func:`__iadd__`          :func:`__missing__`        :func:`__rrshift__`         ..
:func:`__copy__`      :func:`__iand__`          :func:`__mul__`            :func:`__rshift__`          .. 
:func:`__deepcopy__`  :func:`__idivmod__`       :func:`__ne__`             :func:`__rsub__`            ..
:func:`__del__`       :func:`__ifloordiv__`     :func:`__neg__`            :func:`__rtruediv__`        ..
:func:`__delattr__`   :func:`__ilshift__`       :func:`__next__`           :func:`__rxor__`            ..
:func:`__delitem__`   :func:`__imod__`          :func:`__new__`            :func:`__set__`             ..
:func:`__dir__`       :func:`__imul__`          :func:`__or__`             :func:`__setattr__`         ..
:func:`__divmod__`    :func:`__index__`         :func:`__pos__`            :func:`__setitem__`         ..
:func:`__eq__`        :func:`__int__`           :func:`__pow__`            :func:`__setstate__`        ..
:func:`__enter__`     :func:`__invert__`        :func:`__radd__`           :func:`__slots__`           ..
:func:`__exit__`      :func:`__ior__`           :func:`__rand__`           :func:`__str__`             ..
:func:`__format__`    :func:`__ipow__`          :func:`__rdivmod__`        :func:`__sub__`             ..
:func:`__floordiv__`  :func:`__irshift__`       :func:`__reduce__`         :func:`__subclasscheck__`   ..
:func:`__float__`     :func:`__isub__`          :func:`__reduce_ex__`      :func:`__subclasshook__`    ..
:func:`__floor__`     :func:`__iter__`          :func:`__repr__`           :func:`__truediv__`         ..
====================  ========================  =========================  =========================   =================


Basics
******

.. function:: __init__(x)

  :term:`Pyccel omicron`,

.. function:: __repr__

  :term:`Pyccel omicron`,
  :term:`Pyccel beta`,

.. function:: __str__

  :term:`Pyccel omicron`,
  :term:`Pyccel beta`,

.. function:: __bytes__

  :term:`Pyccel restriction`,

.. function:: __format__

  :term:`Pyccel omicron`,
  :term:`Pyccel beta`,


Classes That Act Like Iterators
*******************************

.. function:: __iter__

  :term:`Pyccel omicron`,
  :term:`Pyccel lambda`,

.. function:: __next__

  :term:`Pyccel omicron`,
  :term:`Pyccel lambda`,

.. function:: __reversed__

  :term:`Pyccel omicron`,
  :term:`Pyccel lambda`,


Computed Attributes
*******************

.. function:: __getattribute__

  :term:`Pyccel restriction`,

.. function:: __getattr__

  :term:`Pyccel restriction`,

.. function:: __setattr__ 

  :term:`Pyccel restriction`,

.. function:: __delattr__ 

  :term:`Pyccel restriction`,

.. function:: __dir__

  :term:`Pyccel restriction`,

 
Classes That Act Like Functions
*******************************

.. function:: __call__

  :term:`Pyccel omicron`,


Classes That Act Like Sets
**************************

.. function:: __len__

  :term:`Pyccel beta`,
  :term:`Pyccel lambda`,

.. function:: __contains__

  :term:`Pyccel restriction`,

 
Classes That Act Like Dictionaries
**********************************

.. function:: __getitem__

  :term:`Pyccel restriction`,

.. function:: __setitem__

  :term:`Pyccel restriction`,

.. function:: __delitem__

  :term:`Pyccel restriction`,

.. function:: __missing__

  :term:`Pyccel restriction`,


Classes That Act Like Numbers
*****************************

.. function:: __add__       

  :term:`Pyccel omicron`,

.. function:: __sub__ 

  :term:`Pyccel omicron`,

.. function:: __mul__

  :term:`Pyccel omicron`,

.. function:: __truediv__ 

  :term:`Pyccel omicron`,

.. function:: __floordiv__

  :term:`Pyccel omicron`,

.. function:: __mod__

  :term:`Pyccel omicron`,

.. function:: __divmod__ 

  :term:`Pyccel omicron`,

.. function:: __pow__           

  :term:`Pyccel omicron`,

.. function:: __lshift__      

  :term:`Pyccel omicron`,

.. function:: __rshift__

  :term:`Pyccel omicron`,

.. function:: __and__  

  :term:`Pyccel omicron`,

.. function:: __xor__   

  :term:`Pyccel omicron`,

.. function:: __or__          

  :term:`Pyccel omicron`,



.. function:: __radd__  

  :term:`Pyccel omicron`,

.. function:: __rsub__        

  :term:`Pyccel omicron`,

.. function:: __rmul__   

  :term:`Pyccel omicron`,

.. function:: __rtruediv__    

  :term:`Pyccel omicron`,

.. function:: __rfloordiv__

  :term:`Pyccel omicron`,

.. function:: __rmod__     

  :term:`Pyccel omicron`,

.. function:: __rdivmod__   

  :term:`Pyccel omicron`,

.. function:: __rpow__  

  :term:`Pyccel omicron`,

.. function:: __rlshift__    

  :term:`Pyccel omicron`,

.. function:: __rrshift__

  :term:`Pyccel omicron`,

.. function:: __rand__   

  :term:`Pyccel omicron`,

.. function:: __rxor__      

  :term:`Pyccel omicron`,

.. function:: __ror__        



.. function:: __iadd__  

  :term:`Pyccel omicron`,

.. function:: __isub__   

  :term:`Pyccel omicron`,

.. function:: __imul__   

  :term:`Pyccel omicron`,

.. function:: __itruediv__    

  :term:`Pyccel omicron`,

.. function:: __ifloordiv__

  :term:`Pyccel omicron`,

.. function:: __imod__  

  :term:`Pyccel omicron`,

.. function:: __idivmod__   

  :term:`Pyccel omicron`,

.. function:: __ipow__   

  :term:`Pyccel omicron`,

.. function:: __ilshift__   

  :term:`Pyccel omicron`,

.. function:: __irshift__

  :term:`Pyccel omicron`,

.. function:: __iand__  

  :term:`Pyccel omicron`,

.. function:: __ixor__     

  :term:`Pyccel omicron`,

.. function:: __ior__          

  :term:`Pyccel omicron`,



.. function:: __neg__       

  :term:`Pyccel omicron`,

.. function:: __pos__  

  :term:`Pyccel omicron`,

.. function:: __abs__  

  :term:`Pyccel omicron`,

.. function:: __invert__  

  :term:`Pyccel omicron`,

.. function:: __complex__

  :term:`Pyccel omicron`,

.. function:: __int__ 

  :term:`Pyccel omicron`,

.. function:: __float__ 

  :term:`Pyccel omicron`,

.. function:: __round__ 

  :term:`Pyccel omicron`,

.. function:: __ceil__   

  :term:`Pyccel omicron`,

.. function:: __floor__

  :term:`Pyccel omicron`,

.. function:: __trunc__ 

  :term:`Pyccel omicron`,

.. function:: __index__      

  :term:`Pyccel omicron`,


Classes That Can Be Compared
****************************

.. function:: __eq__   

  :term:`Pyccel omicron`,
  :term:`Pyccel beta`,

.. function:: __ne__   

  :term:`Pyccel omicron`,
  :term:`Pyccel beta`,

.. function:: __lt__   

  :term:`Pyccel omicron`,
  :term:`Pyccel beta`,

.. function:: __le__  

  :term:`Pyccel omicron`,
  :term:`Pyccel beta`,

.. function:: __gt__

  :term:`Pyccel omicron`,
  :term:`Pyccel beta`,

.. function:: __ge__   

  :term:`Pyccel omicron`,
  :term:`Pyccel beta`,

.. function:: __bool__

  :term:`Pyccel omicron`,
  :term:`Pyccel beta`,


Classes That Can Be Serialized
******************************

.. function:: __copy__     

  :term:`Pyccel beta`,

.. function:: __deepcopy__    

  :term:`Pyccel beta`,

.. function:: __getstate__ 

  :term:`Pyccel restriction`,

.. function:: __reduce__ 

  :term:`Pyccel omicron`,
  :term:`Pyccel lambda`,

.. function:: __reduce_ex__

  :term:`Pyccel omicron`,
  :term:`Pyccel lambda`,

.. function:: __setstate__ 

  :term:`Pyccel restriction`,


Classes That Can Be Used in a with Block
****************************************

.. function:: __enter__     

  :term:`Pyccel omicron`,
  :term:`Pyccel lambda`,

.. function:: __exit__

  :term:`Pyccel omicron`,
  :term:`Pyccel lambda`,


Others
******

.. function:: __new__    

  :term:`Pyccel restriction`,

.. function:: __del__  

  :term:`Pyccel omicron`,

.. function:: __slots__  

  :term:`Pyccel restriction`,

.. function:: __hash__            

  :term:`Pyccel restriction`,

.. function:: __get__

  :term:`Pyccel beta`,

.. function:: __set__   

  :term:`Pyccel beta`,

.. function:: __subclasscheck__

  :term:`Pyccel restriction`,

.. function:: __subclasshook__ 

  :term:`Pyccel restriction`,

.. function:: __instancecheck__

  :term:`Pyccel restriction`,



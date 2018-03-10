
.. _built-in-funcs:



Built-in Functions
==================

The Python interpreter has a number of functions built into it that are always
available.  They are listed here in alphabetical order.

===================  =================  ==================  =================  ====================
..                   ..                 Built-in Functions  ..                 ..
===================  =================  ==================  =================  ====================
:func:`abs`          :func:`divmod`     :func:`input`       :func:`open`       :func:`staticmethod`
:func:`all`          :func:`enumerate`  :func:`int`         :func:`ord`        :func:`str`
:func:`any`          :func:`eval`       :func:`isinstance`  :func:`pow`        :func:`sum`
:func:`basestring`   :func:`execfile`   :func:`issubclass`  :func:`print`      :func:`super`
:func:`bin`          :func:`file`       :func:`iter`        :func:`property`   :func:`tuple`
:func:`bool`         :func:`filter`     :func:`len`         :func:`range`      :func:`type`
:func:`bytearray`    :func:`float`      |func-list|_        :func:`raw_input`  :func:`unichr`
:func:`callable`     :func:`format`     :func:`locals`      :func:`reduce`     :func:`unicode`
:func:`chr`          |func-frozenset|_  :func:`long`        :func:`reload`     :func:`vars`
:func:`classmethod`  :func:`getattr`    :func:`map`         |func-repr|_       :func:`xrange`
:func:`cmp`          :func:`globals`    :func:`max`         :func:`reversed`   :func:`zip`
:func:`compile`      :func:`hasattr`    |func-memoryview|_  :func:`round`      :func:`__import__`
:func:`complex`      :func:`hash`       :func:`min`         |func-set|_        ..
:func:`delattr`      :func:`help`       :func:`next`        :func:`setattr`    ..
|func-dict|_         :func:`hex`        :func:`object`      :func:`slice`      ..
:func:`dir`          :func:`id`         :func:`oct`         :func:`sorted`     ..
===================  =================  ==================  =================  ====================

Some of these functions like *abs* are covered in the :term:`Pyccel beta` version, while others like *all* will be covered in the :term:`Pyccel Functional Programming` version. Finally, there are also some functions that are under the :term:`Pyccel restriction` and will not be covered.

.. |func-dict| replace:: ``dict()``
.. |func-frozenset| replace:: ``frozenset()``
.. |func-list| replace:: ``list()``
.. |func-memoryview| replace:: ``memoryview()``
.. |func-repr| replace:: ``repr()``
.. |func-set| replace:: ``set()``

.. function:: abs(x)

  :term:`Pyccel beta`,
  `Python documentation for abs <https://docs.python.org/3/library/functions.html#abs>`_

.. function:: all(x)

  :term:`Pyccel Functional Programming`,
  `Python documentation for all <https://docs.python.org/3/library/functions.html#all>`_

.. function:: any(x)

  :term:`Pyccel Functional Programming`,
  `Python documentation for any <https://docs.python.org/3/library/functions.html#any>`_

.. function:: basestring(x)

  :term:`Pyccel restriction`,
  `Python documentation for basestring <https://docs.python.org/3/library/functions.html#basestring>`_

.. function:: bin(x)

  :term:`Pyccel restriction`,
  `Python documentation for bin <https://docs.python.org/3/library/functions.html#bin>`_

.. function:: bool(x)

  :term:`Pyccel beta`,
  `Python documentation for bool <https://docs.python.org/3/library/functions.html#bool>`_

.. function:: bytearray(x)

  :term:`Pyccel restriction`,
  `Python documentation for bytearray <https://docs.python.org/3/library/functions.html#bytearray>`_

.. function:: callable(object)

  :term:`Pyccel Functional Programming`,
  `Python documentation for callable <https://docs.python.org/3/library/functions.html#callable>`_

.. function:: chr(i)

  :term:`Pyccel beta`,
  `Python documentation for chr <https://docs.python.org/3/library/functions.html#chr>`_

.. function:: classmethod(function)

  :term:`Pyccel Functional Programming`,
  `Python documentation for classmethod <https://docs.python.org/3/library/functions.html#classmethod>`_

.. function:: cmp(x, y)

  :term:`Pyccel beta`,
  `Python documentation for cmp <https://docs.python.org/3/library/functions.html#cmp>`_

.. function:: compile(source, filename, mode[, flags[, dont_inherit]])

  :term:`Pyccel beta`,
  `Python documentation for compile <https://docs.python.org/3/library/functions.html#compile>`_

.. class:: complex([real[, imag]])

  :term:`Pyccel beta`,
  `Python documentation for class <https://docs.python.org/3/library/functions.html#class>`_

.. function:: delattr(object, name)

  :term:`Pyccel restriction`,
  `Python documentation for delattr <https://docs.python.org/3/library/functions.html#delattr>`_

.. _func-dict:
.. class:: dict(\**kwarg)
           dict(mapping, \**kwarg)
           dict(iterable, \**kwarg)
  :noindex:

  :term:`Pyccel restriction`,
  `Python documentation for dict <https://docs.python.org/3/library/functions.html#dict>`_


.. function:: divmod(a, b)

  :term:`Pyccel beta`,
  `Python documentation for divmod <https://docs.python.org/3/library/functions.html#divmod>`_

.. function:: enumerate(sequence, start=0)

  :term:`Pyccel Functional Programming`,
  `Python documentation for enumerate <https://docs.python.org/3/library/functions.html#enumerate>`_

.. function:: eval(expression[, globals[, locals]])

  :term:`Pyccel beta`,
  `Python documentation for eval <https://docs.python.org/3/library/functions.html#eval>`_

.. function:: execfile(filename[, globals[, locals]])

  :term:`Pyccel beta`,
  `Python documentation for execfile <https://docs.python.org/3/library/functions.html#execfile>`_

.. function:: file(name[, mode[, buffering]])

  :term:`Pyccel restriction`,
  `Python documentation for file <https://docs.python.org/3/library/functions.html#file>`_

.. function:: filter(function, iterable)

  :term:`Pyccel Functional Programming`,
  `Python documentation for filter <https://docs.python.org/3/library/functions.html#filter>`_

.. class:: float([x])

  :term:`Pyccel beta`,
  `Python documentation for float <https://docs.python.org/3/library/functions.html#float>`_

.. function:: format(value[, format_spec])


.. _func-frozenset:
.. class:: frozenset([iterable])
   :noindex:


.. function:: getattr(object, name[, default])

.. function:: globals()

.. function:: hasattr(object, name)

.. function:: hash(object)

.. function:: help([object])

.. function:: hex(x)

.. function:: id(object)

.. function:: input([prompt])

.. class:: int(x=0)
           int(x, base=10)

.. function:: isinstance(object, classinfo)

.. function:: issubclass(class, classinfo)

.. function:: iter(o[, sentinel])

.. function:: len(s)

.. _func-list:
.. class:: list([iterable])
   :noindex:

.. class:: long(x=0)
           long(x, base=10)


.. function:: locals()

.. function:: map(function, iterable, ...)


.. function:: max(iterable[, key])
              max(arg1, arg2, \*args[, key])

.. _func-memoryview:
.. function:: memoryview(obj)
   :noindex:


.. function:: min(iterable[, key])
              min(arg1, arg2, \*args[, key])

.. function:: next(iterator[, default])

.. class:: object()

.. function:: oct(x)

.. function:: open(name[, mode[, buffering]])



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

Some of these functions like *abs* are covered in the :term:`Pyccel beta` version, while others like *all* will be covered in the :term:`Pyccel lambda` version. Finally, there are also some functions that are under the :term:`Pyccel restriction` and will not be covered.

.. using :func:`dict` would create a link to another page, so local targets are
   used, with replacement texts to make the output in the table consistent

.. |func-dict| replace:: ``dict()``
.. |func-frozenset| replace:: ``frozenset()``
.. |func-list| replace:: ``list()``
.. |func-memoryview| replace:: ``memoryview()``
.. |func-repr| replace:: ``repr()``
.. |func-set| replace:: ``set()``

.. function:: abs(x)

  :term:`Pyccel alpha`,
  `Python documentation for abs <https://docs.python.org/3/library/functions.html#abs>`_


.. function:: all(x)

  :term:`Pyccel lambda`,
  `Python documentation for all <https://docs.python.org/3/library/functions.html#all>`_


.. function:: any(x)

  :term:`Pyccel lambda`,
  `Python documentation for any <https://docs.python.org/3/library/functions.html#any>`_


.. function:: basestring(x)

  :term:`Pyccel restriction`,
  `Python documentation for basestring <https://docs.python.org/3/library/functions.html#basestring>`_


.. function:: bin(x)

  :term:`Pyccel restriction`,
  `Python documentation for bin <https://docs.python.org/3/library/functions.html#bin>`_


.. function:: bool(x)

  :term:`Pyccel alpha`,
  `Python documentation for bool <https://docs.python.org/3/library/functions.html#bool>`_


.. function:: bytearray(x)

  :term:`Pyccel restriction`,
  `Python documentation for bytearray <https://docs.python.org/3/library/functions.html#bytearray>`_


.. function:: callable(object)

  :term:`Pyccel lambda`,
  :term:`Pyccel omicron`,
  `Python documentation for callable <https://docs.python.org/3/library/functions.html#callable>`_


.. function:: chr(i)

  :term:`Pyccel alpha`,
  `Python documentation for chr <https://docs.python.org/3/library/functions.html#chr>`_


.. function:: classmethod(function)

  :term:`Pyccel omicron`,
  `Python documentation for classmethod <https://docs.python.org/3/library/functions.html#classmethod>`_


.. function:: cmp(x, y)

  :term:`Pyccel beta`,
  `Python documentation for cmp <https://docs.python.org/3/library/functions.html#cmp>`_


.. function:: compile(source, filename, mode[, flags[, dont_inherit]])

  :term:`Pyccel beta`,
  `Python documentation for compile <https://docs.python.org/3/library/functions.html#compile>`_


.. class:: complex([real[, imag]])

  :term:`Pyccel alpha`,
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

  :term:`Pyccel alpha`,
  `Python documentation for divmod <https://docs.python.org/3/library/functions.html#divmod>`_


.. function:: enumerate(sequence, start=0)

  :term:`Pyccel lambda`,
  `Python documentation for enumerate <https://docs.python.org/3/library/functions.html#enumerate>`_


.. function:: eval(expression[, globals[, locals]])

  :term:`Pyccel beta`,
  :term:`Pyccel lambda`,
  `Python documentation for eval <https://docs.python.org/3/library/functions.html#eval>`_


.. function:: execfile(filename[, globals[, locals]])

  :term:`Pyccel beta`,
  `Python documentation for execfile <https://docs.python.org/3/library/functions.html#execfile>`_


.. function:: file(name[, mode[, buffering]])

  :term:`Pyccel restriction`,
  `Python documentation for file <https://docs.python.org/3/library/functions.html#file>`_


.. function:: filter(function, iterable)

  :term:`Pyccel lambda`,
  `Python documentation for filter <https://docs.python.org/3/library/functions.html#filter>`_


.. class:: float([x])

  :term:`Pyccel alpha`,
  `Python documentation for float <https://docs.python.org/3/library/functions.html#float>`_


.. function:: format(value[, format_spec])

  :term:`Pyccel beta`,
  `Python documentation for format <https://docs.python.org/3/library/functions.html#format>`_


.. _func-frozenset:
.. class:: frozenset([iterable])
   :noindex:

  :term:`Pyccel restriction`,
  `Python documentation for frozenset <https://docs.python.org/3/library/functions.html#frozenset>`_


.. function:: getattr(object, name[, default])

  :term:`Pyccel restriction`,
  `Python documentation for file <https://docs.python.org/3/library/functions.html#file>`_


.. function:: globals()

  :term:`Pyccel restriction`,
  `Python documentation for globals <https://docs.python.org/3/library/functions.html#globals>`_


.. function:: hasattr(object, name)

  :term:`Pyccel restriction`,
  `Python documentation for hasattr <https://docs.python.org/3/library/functions.html#hasattr>`_


.. function:: hash(object)

  :term:`Pyccel restriction`,
  `Python documentation for hash <https://docs.python.org/3/library/functions.html#hash>`_


.. function:: help([object])

  :term:`Pyccel restriction`,
  `Python documentation for help <https://docs.python.org/3/library/functions.html#help>`_


.. function:: hex(x)

  :term:`Pyccel restriction`,
  `Python documentation for hex <https://docs.python.org/3/library/functions.html#hex>`_


.. function:: id(object)

  :term:`Pyccel beta`,
  `Python documentation for id <https://docs.python.org/3/library/functions.html#id>`_


.. function:: input([prompt])

  :term:`Pyccel beta`,
  `Python documentation for input <https://docs.python.org/3/library/functions.html#input>`_


.. class:: int(x=0)
           int(x, base=10)

  :term:`Pyccel alpha`,
  `Python documentation for int <https://docs.python.org/3/library/functions.html#int>`_


.. function:: isinstance(object, classinfo)

  :term:`Pyccel omicron`,
  `Python documentation for isinstance <https://docs.python.org/3/library/functions.html#isinstance>`_


.. function:: issubclass(class, classinfo)

  :term:`Pyccel restriction`,
  `Python documentation for issubclass <https://docs.python.org/3/library/functions.html#issubclass>`_


.. function:: iter(o[, sentinel])

  :term:`Pyccel lambda`,
  `Python documentation for iter <https://docs.python.org/3/library/functions.html#iter>`_


.. function:: len(s)

  :term:`Pyccel alpha`,
  `Python documentation for len <https://docs.python.org/3/library/functions.html#len>`_


.. _func-list:
.. class:: list([iterable])
   :noindex:

  :term:`Pyccel alpha`,
  :term:`Pyccel lambda`,
  `Python documentation for list <https://docs.python.org/3/library/functions.html#list>`_


.. class:: long(x=0)
           long(x, base=10)

  :term:`Pyccel beta`,
  `Python documentation for long <https://docs.python.org/3/library/functions.html#long>`_


.. function:: locals()

  :term:`Pyccel restriction`,
  `Python documentation for locals <https://docs.python.org/3/library/functions.html#locals>`_


.. function:: map(function, iterable, ...)

  :term:`Pyccel lambda`,
  `Python documentation for map <https://docs.python.org/3/library/functions.html#map>`_


.. function:: max(iterable[, key])
              max(arg1, arg2, \*args[, key])

  :term:`Pyccel alpha`,
  :term:`Pyccel lambda`,
  `Python documentation for max <https://docs.python.org/3/library/functions.html#max>`_


.. _func-memoryview:
.. function:: memoryview(obj)
   :noindex:

  :term:`Pyccel restriction`,
  `Python documentation for memoryview <https://docs.python.org/3/library/functions.html#memoryview>`_


.. function:: min(iterable[, key])
              min(arg1, arg2, \*args[, key])

  :term:`Pyccel alpha`,
  :term:`Pyccel lambda`,
  `Python documentation for min <https://docs.python.org/3/library/functions.html#min>`_


.. function:: next(iterator[, default])

  :term:`Pyccel lambda`,
  `Python documentation for next <https://docs.python.org/3/library/functions.html#next>`_


.. class:: object()

  :term:`Pyccel beta`,
  :term:`Pyccel omicron`,
  `Python documentation for object <https://docs.python.org/3/library/functions.html#object>`_


.. function:: oct(x)

  :term:`Pyccel restriction`,
  `Python documentation for oct <https://docs.python.org/3/library/functions.html#oct>`_


.. function:: open(name[, mode[, buffering]])

  :term:`Pyccel beta`,
  `Python documentation for open <https://docs.python.org/3/library/functions.html#open>`_


.. function:: ord(c)

  :term:`Pyccel restriction`,
  `Python documentation for ord <https://docs.python.org/3/library/functions.html#ord>`_


.. function:: print(\*objects, sep=' ', end='\\n', file=sys.stdout)

  :term:`Pyccel alpha`,
  `Python documentation for print <https://docs.python.org/3/library/functions.html#print>`_


.. function:: pow(x, y[, z])

  :term:`Pyccel alpha`,
  `Python documentation for pow <https://docs.python.org/3/library/functions.html#pow>`_


.. class:: property([fget[, fset[, fdel[, doc]]]])

  :term:`Pyccel omicron`,
  `Python documentation for property <https://docs.python.org/3/library/functions.html#property>`_


.. function:: range(stop)
              range(start, stop[, step])

  :term:`Pyccel alpha`,
  `Python documentation for range <https://docs.python.org/3/library/functions.html#range>`_

.. function:: raw_input([prompt])

  :term:`Pyccel beta`,
  `Python documentation for raw_input <https://docs.python.org/3/library/functions.html#raw_input>`_


.. function:: reduce(function, iterable[, initializer])

  :term:`Pyccel lambda`,
  `Python documentation for reduce <https://docs.python.org/3/library/functions.html#reduce>`_


.. function:: reload(module)

  :term:`Pyccel restriction`,
  `Python documentation for reload <https://docs.python.org/3/library/functions.html#reload>`_


.. _func-repr:
.. function:: repr(object)

  :term:`Pyccel beta`,
  `Python documentation for repr <https://docs.python.org/3/library/functions.html#repr>`_


.. function:: reversed(seq)

  :term:`Pyccel lambda`,
  `Python documentation for reversed <https://docs.python.org/3/library/functions.html#reversed>`_


.. function:: round(number[, ndigits])

  :term:`Pyccel alpha`,
  `Python documentation for round <https://docs.python.org/3/library/functions.html#round>`_


.. _func-set:
.. class:: set([iterable])
   :noindex:

  :term:`Pyccel lambda`,
  `Python documentation for func-set <https://docs.python.org/3/library/functions.html#func-set>`_


.. function:: setattr(object, name, value)

  :term:`Pyccel restriction`,
  `Python documentation for setattr <https://docs.python.org/3/library/functions.html#setattr>`_


.. class:: slice(stop)
           slice(start, stop[, step])

   .. index:: single: Numerical Python

  :term:`Pyccel alpha`,
  `Python documentation for slice <https://docs.python.org/3/library/functions.html#slice>`_


.. function:: sorted(iterable[, cmp[, key[, reverse]]])

  :term:`Pyccel lambda`,
  `Python documentation for sorted <https://docs.python.org/3/library/functions.html#sorted>`_


.. function:: staticmethod(function)

  :term:`Pyccel omicron`,
  `Python documentation for staticmethod <https://docs.python.org/3/library/functions.html#staticmethod>`_


.. class:: str(object='')

  :term:`Pyccel beta`,
  `Python documentation for str <https://docs.python.org/3/library/functions.html#str>`_


.. function:: sum(iterable[, start])

  :term:`Pyccel lambda`,
  `Python documentation for sum <https://docs.python.org/3/library/functions.html#sum>`_


.. function:: super(type[, object-or-type])

  :term:`Pyccel omicron`,
  `Python documentation for super <https://docs.python.org/3/library/functions.html#super>`_


.. function:: tuple([iterable])

  :term:`Pyccel alpha`,
  :term:`Pyccel lambda`,
  `Python documentation for tuple <https://docs.python.org/3/library/functions.html#tuple>`_


.. class:: type(object)
           type(name, bases, dict)

   .. index:: object: type

  :term:`Pyccel omicron`,
  `Python documentation for type <https://docs.python.org/3/library/functions.html#type>`_


.. function:: unichr(i)

  :term:`Pyccel restriction`,
  `Python documentation for unichr <https://docs.python.org/3/library/functions.html#unichr>`_


.. function:: unicode(object='')
              unicode(object[, encoding [, errors]])

  :term:`Pyccel restriction`,
  `Python documentation for unicode <https://docs.python.org/3/library/functions.html#unicode>`_


.. function:: vars([object])

  :term:`Pyccel restriction`,
  `Python documentation for vars <https://docs.python.org/3/library/functions.html#vars>`_


.. function:: xrange(stop)
              xrange(start, stop[, step])

  :term:`Pyccel alpha`,
  `Python documentation for xrange <https://docs.python.org/3/library/functions.html#xrange>`_


.. function:: zip([iterable, ...])

  :term:`Pyccel lambda`,
  `Python documentation for zip <https://docs.python.org/3/library/functions.html#zip>`_


.. function:: __import__(name[, globals[, locals[, fromlist[, level]]]])

  :term:`Pyccel restriction`,
  `Python documentation for __import__ <https://docs.python.org/3/library/functions.html#__import__>`_


.. _non-essential-built-in-funcs:

Non-essential Built-in Functions
================================

There are several built-in functions that are no longer essential to learn, know
or use in modern Python programming.  They have been kept here to maintain
backwards compatibility with programs written for older versions of Python.

Python programmers, trainers, students and book writers should feel free to
bypass these functions without concerns about missing something important.


.. function:: apply(function, args[, keywords])

  :term:`Pyccel restriction`,
  `Python documentation for apply <https://docs.python.org/3/library/functions.html#apply>`_


.. function:: buffer(object[, offset[, size]])

  :term:`Pyccel restriction`,
  `Python documentation for buffer <https://docs.python.org/3/library/functions.html#buffer>`_


.. function:: coerce(x, y)

  :term:`Pyccel restriction`,
  `Python documentation for coercive <https://docs.python.org/3/library/functions.html#coercive>`_


.. function:: intern(string)

  :term:`Pyccel restriction`,
  `Python documentation for intern <https://docs.python.org/3/library/functions.html#intern>`_



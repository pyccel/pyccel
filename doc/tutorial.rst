.. highlight:: rst

First Steps with Pyccel
=======================

This document is meant to give a tutorial-like overview Pyccel usage.

The green arrows designate "more info" links leading to advanced sections about
the described task.


Install Pyccel
**************

Install Pyccel from a distribution package with ::

  $ python setup.py install --prefix=MY_INSTALL_PATH


Setting up the low-level code sources
*************************************

The root directory of a Pyccel collection of pyccel sources
is called the :term:`source directory`.  This directory also contains the Pyccel
configuration file :file:`conf.py`, where you can configure all aspects of how
Pyccel reads your sources and builds your project.  [#]_

Pyccel comes with a script called :program:`pyccel-quickstart` that sets up a
source directory and creates a default :file:`conf.py` with the most useful
configuration values from a few questions it asks you. Just run ::

   $ pyccel-quickstart -h

for help.

Defining document structure
***************************

Let's assume you've run :program:`pyccel-quickstart` for a project **poisson**.  It created a source
directory with :file:`conf.py` and a directory **poisson** that contains a master file, :file:`main.py` (if you used the defaults).  The main function of the :term:`master document` is to
serve as an example of a **main program**.

Adding content
**************

In Pyccel source files, you can use most features of standard reStructuredText.
There are also several features added by Pyccel.  For example, you can add
cross-file references in a portable way (which works for all output types) using
the :rst:role:`ref` role.

For an example, if you are viewing the HTML version you can look at the source
for this document -- use the "Show Source" link in the sidebar.


Running the build
*****************

Now that you have added some files and content, let's make a first build of the
docs.  A build is started with the :program:`pyccel-build` program, called like
this::

   $ pyccel-build -b html sourcedir builddir

where *sourcedir* is the :term:`source directory`, and *builddir* is the
directory in which you want to place the built documentation.
The :option:`-b <pyccel-build -b>` option selects a builder; in this example
Pyccel will build HTML files.

|more| Refer to the :program:`pyccel-build man page <pyccel-build>` for all
options that :program:`pyccel-build` supports.

However, :program:`pyccel-quickstart` script creates a :file:`Makefile` and a
:file:`make.bat` which make life even easier for you:  with them you only need
to run ::

   $ make html

to build HTML docs in the build directory you chose.  Execute ``make`` without
an argument to see which targets are available.

.. admonition:: How do I generate PDF documents?

   ``make latexpdf`` runs the :mod:`LaTeX builder
   <pyccel.builders.latex.LaTeXBuilder>` and readily invokes the pdfTeX
   toolchain for you.


Documenting objects
*******************

One of Pyccel's main objectives is easy documentation of :dfn:`objects` (in a
very general sense) in any :dfn:`domain`.  A domain is a collection of object
types that belong together, complete with markup to create and reference
descriptions of these objects.

The most prominent domain is the Python domain. For example, to document
Python's built-in function ``enumerate()``, you would add this to one of your
source files::

   .. py:function:: enumerate(sequence[, start=0])

      Return an iterator that yields tuples of an index and an item of the
      *sequence*. (And so on.)

This is rendered like this:

.. py:function:: enumerate(sequence[, start=0])

   Return an iterator that yields tuples of an index and an item of the
   *sequence*. (And so on.)

The argument of the directive is the :dfn:`signature` of the object you
describe, the content is the documentation for it.  Multiple signatures can be
given, each in its own line.

The Python domain also happens to be the default domain, so you don't need to
prefix the markup with the domain name::

   .. function:: enumerate(sequence[, start=0])

      ...

does the same job if you keep the default setting for the default domain.

There are several more directives for documenting other types of Python objects,
for example :rst:dir:`py:class` or :rst:dir:`py:method`.  There is also a
cross-referencing :dfn:`role` for each of these object types.  This markup will
create a link to the documentation of ``enumerate()``::

   The :py:func:`enumerate` function can be used for ...

And here is the proof: A link to :func:`enumerate`.

Again, the ``py:`` can be left out if the Python domain is the default one.  It
doesn't matter which file contains the actual documentation for ``enumerate()``;
Pyccel will find it and create a link to it.

Each domain will have special rules for how the signatures can look like, and
make the formatted output look pretty, or add specific features like links to
parameter types, e.g. in the C/C++ domains.

|more| See :ref:`domains` for all the available domains and their
directives/roles.


Basic configuration
*******************

Earlier we mentioned that the :file:`conf.py` file controls how Pyccel processes
your documents.  In that file, which is executed as a Python source file, you
assign configuration values.  For advanced users: since it is executed by
Pyccel, you can do non-trivial tasks in it, like extending :data:`sys.path` or
importing a module to find out the version you are documenting.

The config values that you probably want to change are already put into the
:file:`conf.py` by :program:`pyccel-quickstart` and initially commented out
(with standard Python syntax: a ``#`` comments the rest of the line).  To change
the default value, remove the hash sign and modify the value.  To customize a
config value that is not automatically added by :program:`pyccel-quickstart`,
just add an additional assignment.

Keep in mind that the file uses Python syntax for strings, numbers, lists and so
on.  The file is saved in UTF-8 by default, as indicated by the encoding
declaration in the first line.  If you use non-ASCII characters in any string
value, you need to use Python Unicode strings (like ``project = u'Exposé'``).

..  |more| See :ref:`build-config` for documentation of all available config values.


More topics to be covered
-------------------------

- :doc:`Other extensions <extensions>`:

  * :doc:`ext/math`,
  * ...


.. rubric:: Footnotes

.. [#] This is the usual layout.  However, :file:`conf.py` can also live in
       another directory, the :term:`configuration directory`.  Refer to the
       :program:`pyccel-build man page <pyccel-build>` for more information.

.. |more| image:: more.png
          :align: middle
          :alt: more info

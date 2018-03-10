.. _specs:

The Pyccel Compiler
===================

Compiling a single file
***********************

TODO

Syntax analysis
^^^^^^^^^^^^^^^

Semantic analysis
^^^^^^^^^^^^^^^^^

Code generation
^^^^^^^^^^^^^^^

Backend compilation
^^^^^^^^^^^^^^^^^^^

Setting up a project
********************

The root directory of a Pyccel collection of pyccel sources
is called the :term:`source directory`.  This directory also contains the Pyccel
configuration file :file:`conf.py`, where you can configure all aspects of how
Pyccel converts your sources and builds your project. 

Pyccel comes with a script called :program:`pyccel-quickstart` that sets up a
source directory and creates a default :file:`conf.py` with the most useful
configuration values. Just run ::

   $ pyccel-quickstart -h

for help.

For example, runing::

   $ pyccel-quickstart poisson

will create a directory **poisson** where you will find, inside it:

.. figure:: ../include/pyccel-quickstart_poisson.png 
   :align: center
   :scale: 100% 

   Structure of the **poisson** project after running :program:`pyccel-quickstart`.


Defining document structure
***************************

Let's assume you've run :program:`pyccel-quickstart` for a project **poisson**.  It created a source
directory with :file:`conf.py` and a directory **poisson** that contains a master file, :file:`main.py` (if you used the defaults settings). The main function of the :term:`master document` is to
serve as an example of a **main program**.

Adding content
**************

In Pyccel source files, you can use most features of standard *Python* instructions.
There are also several features added by Pyccel.  For example, you can use multi-threading or distributed memory programing paradigms, as part of the Pyccel language itself.

Running the build
*****************

Now that you have added some files and content, let's make a first build of the
project.  A build is started with the :program:`pyccel-build` program, called like
this::

   $ pyccel-build application 

where *application* is the :term:`application directory` you want to build.

|more| Refer to the :program:`pyccel-build man page <pyccel-build>` for all
options that :program:`pyccel-build` supports.

Notice that :program:`pyccel-quickstart` script creates a build directory :term:`build directory` in which you can use **cmake** or :file:`Makefile`. 
In order to compile *manualy* your project, you just need to go to this build directory and run ::

   $ make

Basic configuration
*******************

.. todo:: add basic configurations.

.. Earlier we mentioned that the :file:`conf.py` file controls how Pyccel processes
.. your documents.  In that file, which is executed as a Python source file, you
.. assign configuration values.  For advanced users: since it is executed by
.. Pyccel, you can do non-trivial tasks in it, like extending :data:`sys.path` or
.. importing a module to find out the version you are documenting.
.. 
.. The config values that you probably want to change are already put into the
.. :file:`conf.py` by :program:`pyccel-quickstart` and initially commented out
.. (with standard Python syntax: a ``#`` comments the rest of the line).  To change
.. the default value, remove the hash sign and modify the value.  To customize a
.. config value that is not automatically added by :program:`pyccel-quickstart`,
.. just add an additional assignment.
.. 
.. Keep in mind that the file uses Python syntax for strings, numbers, lists and so
.. on.  The file is saved in UTF-8 by default, as indicated by the encoding
.. declaration in the first line.  If you use non-ASCII characters in any string
.. value, you need to use Python Unicode strings (like ``project = u'Exposé'``).
.. 
.. ..  |more| See :ref:`build-config` for documentation of all available config values.


Contents
********

.. toctree::
   :maxdepth: 1 

   syntax
   semantic
   codegen
   project
   rules


.. rubric:: Footnotes

.. |more| image:: ../include/more.png
          :align: middle
          :alt: more info


.. default-role:: any

.. _invocation:

Command lines
*************

Invocation of pyccel-quickstart
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :program:`pyccel-quickstart` script generates a Pyccel documentation set.
It is called like this:

.. code-block:: console

   $ pyccel-quickstart [options] [projectdir]

where *projectdir* is the Pyccel documentation set directory in which you want
to place. If you omit *projectdir*, files are generated into current directory
by default.

The :program:`pyccel-quickstart` script has several options:

.. program:: pyccel-quickstart

.. option:: -q, --quiet

   Quiet mode that will skips interactive wizard to specify options.
   This option requires `-a` and `-v` options.

.. option:: -h, --help, --version

   Display usage summary or Pyccel version.

Structure options
_________________

.. option:: --sep

   If specified, separate source and build directories.

.. option:: --dot=DOT

   You can define a prefix for the temporary directories: build, etc
   You can enter another prefix (such as ".") to
   replace the underscore.

Project basic options:
______________________

.. option:: -a AUTHOR, --author=AUTHOR

   Author names. (see :confval:`copyright`).

.. option:: -v VERSION

   Version of project. (see :confval:`version`).

.. option:: -r RELEASE, --release=RELEASE

   Release of project. (see :confval:`release`).

.. option:: -l LANGUAGE, --language=LANGUAGE

   Low-level language. (see :confval:`language`).

.. option:: --suffix-library=SUFFIX_LIBRARY

   Suffix of 3 letters for the project. (see :confval:`source_suffix`).

.. option:: --master=MASTER

   Master file name. (see :confval:`master_doc`).

.. option:: --compiler=COMPILER

   A valid compiler. (see :confval:`compiler_doc`).

.. option:: --include INCLUDE

   path to include directory. (see :confval:`compiler_doc`).

.. option:: --libdir LIBDIR

   path to lib directory. (see :confval:`compiler_doc`).

.. option:: --libs LIBS

   list of libraries to link with. (see :confval:`compiler_doc`).

.. option:: --convert-only

   Converts pyccel files only without build. (see :confval:`convertion_doc`).

Extension options
_________________

.. option:: --ext-blas

   Enable `pyccelext.blas` extension.

.. option:: --ext-math

   Enable `pyccelext.math` extension.

Invocation of pyccel-build
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :program:`pyccel-build` script builds a Pyccel documentation set.  It is
called like this:

.. code-block:: console

   $ pyccel-build [options] sourcedir [filenames]

where *sourcedir* is the :term:`source directory`. Most of the time, 
you don't need to specify any *filenames*.

The :program:`pyccel-build` script has several options:

.. program:: pyccel-build

.. option:: -h, --help, --version

   Display usage summary or Pyccel version.

General options
_______________

.. option:: --output-dir OUTPUT_DIR

   Output directory.

.. option:: --convert-only

   Converts pyccel files only without build. (see :confval:`convertion_doc`).

.. option::   -b BUILDER 

   builder to use (default: fortran)

.. option::   -a   

   write all files (default: only write new and changed files)

.. option::   -E   

   don't use a saved environment, always read all files

.. option::   -j N 

   build in parallel with N processes where possible

Build configuration options
___________________________

.. option::   -c PATH 

   path where configuration file (conf.py) is located (default: same as SOURCEDIR)

.. option::   -D setting=value

   override a setting in configuration file

Console output options
______________________

.. option:: -v

   increase verbosity (can be repeated)

.. option:: -q 

   no output on stdout, just warnings on stderr

.. option:: -Q 

   no output at all, not even warnings

.. option:: -W 

   turn warnings into errors

pyccel-build
============

The :program:`pyccel-build` script builds a Pyccel documentation set.  It is called like this:

.. code-block:: console

   $ pyccel-build [options] sourcedir [filenames]

where *sourcedir* is the :term:`source directory`. Most of the time, 
you don't need to specify any *filenames*.

The :program:`pyccel-build` script has several options:

.. program:: pyccel-build

.. option:: -h, --help, --version

   Display usage summary or Pyccel version.

General options
***************

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
***************************

.. option::   -c PATH 

   path where configuration file (conf.py) is located (default: same as SOURCEDIR)

.. option::   -D setting=value

   override a setting in configuration file

Console output options
**********************

.. option:: -v

   increase verbosity (can be repeated)

.. option:: -q 

   no output on stdout, just warnings on stderr

.. option:: -Q 

   no output at all, not even warnings

.. option:: -W 

   turn warnings into errors

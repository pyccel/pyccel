.. _glossary:

Glossary
========

.. glossary::

   configuration directory
      The directory containing :file:`conf.py`.  By default, this is the same as
      the :term:`source directory`, but can be set differently with the **-c**
      command-line option.

   directive
      A reStructuredText markup element that allows marking a block of content
      with special meaning.  Directives are supplied not only by docutils, but
      Sphinx and custom extensions can add their own.  

   document name
      Since reST source files can have different extensions (some people like
      ``.txt``, some like ``.rst`` -- the extension can be configured with
      :confval:`source_suffix`) and different OSes have different path
      separators, Sphinx abstracts them: :dfn:`document names` are always
      relative to the :term:`source directory`, the extension is stripped, and
      path separators are converted to slashes.  All values, parameters and such
      referring to "documents" expect such document names.

      Examples for document names are ``index``, ``library/zipfile``, or
      ``reference/datamodel/types``.  Note that there is no leading or trailing
      slash.

   environment
      A structure where information about all documents under the root is saved,
      and used for cross-referencing.  The environment is pickled after the
      parsing stage, so that successive runs only need to read and parse new and
      changed documents.

   master document
      The document that contains the root :rst:dir:`toctree` directive.

   object
      The basic building block of Sphinx documentation.  Every "object
      directive" (e.g. :rst:dir:`function` or :rst:dir:`object`) creates such a
      block; and most objects can be cross-referenced to.

   role
      A reStructuredText markup element that allows marking a piece of text.

   source directory
      The directory which, including its subdirectories, contains all source
      files for one Sphinx project.

   application directory
      The directory which contains Pyccel sources, as a package.

   build directory
      The build directory as you may specify it for cmake.

   Pyccel alpha 
      Pyccel alpha version

   Pyccel beta 
      Pyccel beta release version

   Pyccel omicron 
      Pyccel release version for OOP

   Pyccel lambda 
      Pyccel release version for Functional Programming

   Pyccel restriction
      Denotes a restriction of Python by Pyccel

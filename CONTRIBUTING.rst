.. highlight:: console

Pyccel Developer's Guide
========================

.. topic:: Abstract

   This document describes the development process of Pyccel.

.. contents::
   :local:

The Pyccel source code is managed using Git and is hosted on GitHub.

    git clone git://github.com/ratnania/pyccel

.. .. rubric:: Community
.. 
.. pyccel-users <pyccel-users@googlegroups.com>
..     Mailing list for user support.
.. 
.. pyccel-dev <pyccel-dev@googlegroups.com>
..     Mailing list for development related discussions.
.. 
.. #pyccel-doc on irc.freenode.net
..     IRC channel for development questions and user support.


Bug Reports and Feature Requests
--------------------------------

If you have encountered a problem with Pyccel or have an idea for a new
feature, please submit it to the `issue tracker`_ on GitHub or discuss it
on the pyccel-dev mailing list.

For bug reports, please include the output produced during the build process
and also the log file Pyccel creates after it encounters an un-handled
exception.  The location of this file should be shown towards the end of the
error message.

Including or providing a link to the source files involved may help us fix the
issue.  If possible, try to create a minimal project that produces the error
and post that instead.

.. _`issue tracker`: https://github.com/ratnania/pyccel/issues


Contributing to Pyccel
----------------------

The recommended way for new contributors to submit code to Pyccel is to fork
the repository on GitHub and then submit a pull request after
committing the changes.  The pull request will then need to be approved by one
of the core developers before it is merged into the main repository.

#. Check for open issues or open a fresh issue to start a discussion around a
   feature idea or a bug.
#. If you feel uncomfortable or uncertain about an issue or your changes, feel
   free to email pyccel-dev@googlegroups.com.
#. Fork `the repository`_ on GitHub to start making your changes to the
   **master** branch for next major version, or **stable** branch for next
   minor version.
#. Write a test which shows that the bug was fixed or that the feature works
   as expected.
#. Send a pull request and bug the maintainer until it gets merged and
   published. Make sure to add yourself to AUTHORS_ and the change to
   CHANGES_.

.. _`the repository`: https://github.com/ratnania/pyccel
.. _AUTHORS: https://github.com/ratnania/pyccel/blob/master/AUTHORS
.. _CHANGES: https://github.com/ratnania/pyccel/blob/master/CHANGES


Getting Started
~~~~~~~~~~~~~~~

These are the basic steps needed to start developing on Pyccel.

#. Create an account on GitHub.

#. Fork the main Pyccel repository (`ratnania/pyccel
   <https://github.com/ratnania/pyccel>`_) using the GitHub interface.

#. Clone the forked repository to your machine. ::

       git clone https://github.com/USERNAME/pyccel
       cd pyccel

#. Checkout the appropriate branch.

   For changes that should be included in the next minor release (namely bug
   fixes), use the ``stable`` branch. ::

       git checkout stable

   For new features or other substantial changes that should wait until the
   next major release, use the ``master`` branch.

.. #. Optional: setup a virtual environment. ::
.. 
..        virtualenv ~/pyccelenv
..        . ~/pyccelenv/bin/activate
..        pip install -e .
.. 
.. #. Create a new working branch.  Choose any name you like. ::
.. 
..        git checkout -b feature-xyz
.. 
.. #. Hack, hack, hack.
.. 
..    For tips on working with the code, see the `Coding Guide`_.
.. 
.. #. Test, test, test.  Possible steps:
.. 
..    * Run the unit tests::
.. 
..        pip install .[test,websupport]
..        make test
.. 
..    * Again, it's useful to turn on deprecation warnings on so they're shown in
..      the test output::
.. 
..        PYTHONWARNINGS=all make test
.. 
..    * Arguments to pytest can be passed via tox, e.g. in order to run a
..      particular test::
.. 
..        tox -e py27 tests/test_module.py::test_new_feature
.. 
..    * Build the documentation and check the output for different builders::
.. 
..        make docs target="clean html latexpdf"
.. 
..    * Run code style checks and type checks (type checks require mypy)::
.. 
..        make style-check
..        make type-check
.. 
..    * Run the unit tests under different Python environments using
..      :program:`tox`::
.. 
..        pip install tox
..        tox -v
.. 
..    * Add a new unit test in the ``tests`` directory if you can.
.. 
..    * For bug fixes, first add a test that fails without your changes and passes
..      after they are applied.
.. 
..    * Tests that need a pyccel-build run should be integrated in one of the
..      existing test modules if possible.  New tests that to ``@with_app`` and
..      then ``build_all`` for a few assertions are not good since *the test suite
..      should not take more than a minute to run*.
.. 
.. #. Please add a bullet point to :file:`CHANGES` if the fix or feature is not
..    trivial (small doc updates, typo fixes).  Then commit::
.. 
..        git commit -m '#42: Add useful new feature that does this.'
.. 
..    GitHub recognizes certain phrases that can be used to automatically
..    update the issue tracker.
.. 
..    For example::
.. 
..        git commit -m 'Closes #42: Fix invalid markup in docstring of Foo.bar.'
.. 
..    would close issue #42.
.. 
.. #. Push changes in the branch to your forked repository on GitHub. ::
.. 
..        git push origin feature-xyz
.. 
.. #. Submit a pull request from your branch to the respective branch (``master``
..    or ``stable``) on ``ratnania/pyccel`` using the GitHub interface.
.. 
.. #. Wait for a core developer to review your changes.


Core Developers
~~~~~~~~~~~~~~~

The core developers of Pyccel have write access to the main repository.  They
can commit changes, accept/reject pull requests, and manage items on the issue
tracker.

You do not need to be a core developer or have write access to be involved in
the development of Pyccel.  You can submit patches or create pull requests
from forked repositories and have a core developer add the changes for you.

The following are some general guidelines for core developers:

* Questionable or extensive changes should be submitted as a pull request
  instead of being committed directly to the main repository.  The pull
  request should be reviewed by another core developer before it is merged.

* Trivial changes can be committed directly but be sure to keep the repository
  in a good working state and that all tests pass before pushing your changes.

* When committing code written by someone else, please attribute the original
  author in the commit message and any relevant :file:`CHANGES` entry.


Coding Guide
------------

* Try to use the same code style as used in the rest of the project. 

* For non-trivial changes, please update the :file:`CHANGES` file.  If your
  changes alter existing behavior, please document this.

* New features should be documented.  Include examples and use cases where
  appropriate.  If possible, include a sample that is displayed in the
  generated output.

* When adding a new configuration variable, be sure to document it and update
  :file:`doc/man/pyccel-quickstart.rst` (for example) if it's important enough.

* Add appropriate unit tests.


Debugging Tips
~~~~~~~~~~~~~~

* .

Unit Testing
------------

Pyccel has been tested with pytest runner. Pyccel developers write unit tests
using pytest notation. Utility functions and pytest fixtures for testing are
provided in ``pyccel.testing``. If you are a developer of Pyccel extensions,
you can write unit tests with using pytest. At this time, ``pyccel.testing``
will help your test implementation.

How to use pytest fixtures that are provided by ``pyccel.testing``?
You can require ``'pyccel.testing.fixtures'`` in your test modules or
``conftest.py`` files like this::

   pytest_plugins = 'pyccel.testing.fixtures'

If you want to know more detailed usage, please refer to ``tests/conftest.py``
and other ``test_*.py`` files under ``tests`` directory.

.. note::

   Prior to Pyccel - 1.5.2, Pyccel was running the test with nose.

.. versionadded:: 1.6
   ``pyccel.testing`` as a experimental.

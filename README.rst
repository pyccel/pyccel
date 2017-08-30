pyccel
======

**Pyccel** stands for Python extension language using accelerators.

The aim of **Pyccel** is to provide a simple way to generate automatically, parallel low level code. The main uses would be:

1. Convert a *Python* code into a low-level target language (Fortran or C)

2. Accelerate a selected *Python* code, by using a pre-process, where the original code is translated into a low-level language, then compiled using **f2py** for example.

In order to achieve these tasks, in **Pyccel** we deal with the following points:

a. Implement a new *Python* parser (we do not need to cover all *Python* grammar)

b. Enrich *Python* with new statments to provide multi-threading (although some of them already exist) at the target level

c. Extends the concepts presented in **sympy** allowing for automatic code generation.  

Install
*******

run::

  python setup.py install --prefix=MY_INSTALL_PATH

this will install a *python* library **pyccel** and a *binary* called **pyccel**.

If **prefix** is not given, you will need to be in *sudo* mode. Otherwise, you will need to update your *.bashrc* or *.bash_profile* file with::

  export PYTHONPATH=MY_INSTALL_PATH/lib/python2.7/site-packages/:$PYTHONPATH
  export PATH=MY_INSTALL_PATH/bin:$PATH

for tests, run::

  cd tests
  python run_tests.py

Documentation
*************

Can be found `here <http://ratnani.org/documentations/pyccel/>`_

Examples
********

Hello World
^^^^^^^^^^^

Let us consider the following *Python* file (*helloworld.py*)

.. code-block:: python

  def helloworld():
      print("* Hello World!!")

Now, run the command::

  pyccel --language="fortran" --compiler="gfortran" --filename=helloworld.py

The generated *Fortran* code is

.. code-block:: fortran

  module pyccel_m_helloworld

  implicit none

  contains
  ! ........................................
  subroutine helloworld()
  implicit none

  print *, '* Hello World!!'

  end subroutine
  ! ........................................


  end module pyccel_m_helloworld


BUGS
****

TODO
****


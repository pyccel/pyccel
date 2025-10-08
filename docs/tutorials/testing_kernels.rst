"""""""""""""""""""""""""""""""""""""""""""""""""""""
Prototyping Compiled Functions with Inlined Callables
"""""""""""""""""""""""""""""""""""""""""""""""""""""

============
Introduction
============

When developing new numerical methods, it is often important to test them in a high-dimensional setting to properly evaluate their performance. However, such high-dimensional environments can lead to very slow evaluation times, especially if the expressions being tested cannot be easily vectorised. Pyccel is a useful tool for accelerating such cases.

A common challenge during development is that only a single line in the numerical method, for example the formula for an integrand, needs to change repeatedly between tests. Rewriting the entire function each time in order to recompile can be tedious.

Pyccel’s context functions allow you to pass in user-defined Python callables that get inlined directly into the compiled code for maximum performance. This allows you to continue developing in an interactive environment without rewriting the entire function repeatedly.

In this tutorial, we'll show how to write a general-purpose integration function that can be tested with different kernels. We'll use a simple Python lambda to represent the kernel, which will be embedded into the generated code to avoid any Python overhead during execution.

============================================
Integrating User-Defined Kernels with Pyccel
============================================

Here’s a simple example of how you can write a general-purpose 2D integrator in Python, then compile it with Pyccel while injecting an arbitrary kernel function at compile time.

---------------------------
General integration routine
---------------------------

.. literalinclude:: ./testing_kernels.py
   :language: python
   :linenos:
   :start-after: # MIDPOINT_RULE
   :end-before: # END_MIDPOINT_RULE

Here we have tested the integration of multiple expressions (seen in comments) but we can see that (especially in an interactive environment) it is simpler to use a Python function to specify what should be integrated during a testing phase.

---------------------
Compiling with Pyccel
---------------------

If we define a free function or a lambda function with the expected name (`test_func`) we can now use `epyccel` to get a compiled version of the general integration routine, specific to this test kernel:

.. literalinclude:: ./testing_kernels.py
   :language: python
   :linenos:
   :start-after: # COMPILE
   :end-before: # END_COMPILE

-----
Usage
-----

The compiled method can be used exactly as the original method was used:

.. literalinclude:: ./testing_kernels.py
   :language: python
   :linenos:
   :start-after: # TEST
   :end-before: # END_TEST

--------------
Generated code
--------------

Using a lambda function (or a function with the [`@inline` decorator](./decorators#inline)) for the kernel ensures that the method is inlined. For example the Pyccel-generated translation created by the call above is:

.. code-block:: fortran
   :linenos:

    result_0001 = 0.0_f64
    do i = 0_i64, nx - 1_i64
      do j = 0_i64, ny - 1_i64
        Dummy_0000 = exp(-(xs(i) ** 2_i64 + ys(j) ** 2_i64))
        result_0001 = result_0001 + Dummy_0000 * dx * dy
      end do
    end do

This makes the resulting code faster but it means that `epyccel` will need to be called again to get an updated version if the lambda function `test_func` is modified.
On the other hand this means that multiple versions of the accelerated function co-exist and are usable simultaneously:

.. literalinclude:: ./testing_kernels.py
   :language: python
   :linenos:
   :start-after: # MULTIPLE_TESTS
   :end-before: # END_MULTIPLE_TESTS

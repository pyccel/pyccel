Code Analysis
=============

Complexity
**********

We consider the following matrix-matrix product example

.. code-block:: python

  n = int()
  n = 2

  x = zeros(shape=(n,n), dtype=float)
  y = zeros(shape=(n,n), dtype=float)
  z = zeros(shape=(n,n), dtype=float)

  for i in range(0, n):
      for j in range(0, n):
          for k in range(0, n):
              z[i,j] = z[i,j] + x[i,k]*y[k,j]

now let's run the following command::

  $ pyccel --filename=tests/complexity/inputs/ex3.py --analysis
  arithmetic cost         ~ n**3*(ADD + MUL)
  memory cost             ~ WRITE + n**3*(3*READ + WRITE)
  computational intensity ~ (ADD + MUL)/(3*READ + WRITE)

Let's now consider the following block version of the matrix-matrix product

.. code-block:: python

  n = int()
  b = int()
  m = int()
  n = 10
  b = 2
  p = n / b

  x = zeros(shape=(n,n), dtype=float)
  y = zeros(shape=(n,n), dtype=float)
  z = zeros(shape=(n,n), dtype=float)

  r = zeros(shape=(b,b), dtype=float)
  u = zeros(shape=(b,b), dtype=float)
  v = zeros(shape=(b,b), dtype=float)

  for i in range(0, p):
      for j in range(0, p):
          for k1 in range(0, b):
              for k2 in range(0, b):
                  r[k1,k2] = z[i+k1,j+k2]
          for k in range(0, p):
              for k1 in range(0, b):
                  for k2 in range(0, b):
                      u[k1,k2] = x[i+k1,k+k2]
                      v[k1,k2] = y[k+k1,j+k2]
              for ii in range(0, b):
                  for jj in range(0, b):
                      for kk in range(0, b):
                          r[ii,jj] = r[ii,jj] + u[ii,kk]*v[kk,jj]
          for k1 in range(0, b):
              for k2 in range(0, b):
                  z[i+k1,j+k2] = r[k1,k2]

the analysis is done again using::

  $ pyccel --filename=tests/complexity/inputs/ex4.py --analysis
  arithmetic cost         ~ DIV + b**3*p**3*(ADD + MUL)
  memory cost             ~ 2*READ + 3*WRITE + b**2*p**2*(2*READ + 2*WRITE + p*(2*READ + 2*WRITE + b*(3*READ + WRITE)))
  computational intensity ~ (ADD + MUL)/(3*READ + WRITE)

Now, let us assume we have two level of memories, the **fast** memory represents the **L2** cache. By giving the variables that live in the cache, using **local_vars**, the analysis gives::

  $ pyccel --filename=tests/complexity/inputs/ex4.py --analysis --local_vars="u,v,r"
  arithmetic cost         ~ DIV + b**3*p**3*(ADD + MUL)
  memory cost             ~ 2*READ + 3*WRITE + b**2*p**2*(2*READ*p + READ + WRITE)
  computational intensity ~ b*(ADD + MUL)/(2*READ)

As we can see, the computational intensity is now a linear function of the block size :math:`b`. Therefor, this algorithm will take more advantage of the spatial locality of data.

.. todo:: remove local_vars from **pyccel** command line and use Annotated Comments instead.

.. todo:: for the moment, we only cover the **for** statement. Further work must be done for **if** and **while** statements.

.. todo:: add probability law for the **if** statement.

.. todo:: how to handle the **while** statement?

Arithmetic
^^^^^^^^^^

.. TODO:: add Fusion Mul-Add (FMA) instruction

.. TODO:: add table of costs for all instructions

Memory
^^^^^^

We describe here our *Memory model*. It follows the work of J. Demmel and his collaborators on the matrix multiplication. More details can be found in `J. Demmel's talk`_

.. _J. Demmel's talk: https://people.eecs.berkeley.edu/~demmel/cs267_Spr99/Lectures/Lect_02_1999b.pdf

Here are our assumptions:

1. Two levels of memory: *fast* and *slow*
2. All data are initially in *slow* memory


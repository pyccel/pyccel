Overview
========

.. todo:: add diagram

.. .. tikz:: Constructing the AST for pure python code (no OpenMP/OpenACC). 
.. 
..   \node[draw=black, rectangle, fill=red!40] (fst)  
..   at (0,0)  {FST};
.. 
..   \node at (0.5,0) [color=gray,above=3mm,right=0mm,font=\fontsize{10}{10.2}] {syntax};
..   \node at (0.5,0) [color=gray,below=3mm,right=0mm,font=\fontsize{10}{10.2}] {analysis};
.. 
..   \node[draw=black, rectangle, fill=red!20, font=\fontsize{10}{10.2}] (ast1)  
..   at (3,0)  {AST};
.. 
..   \node at (3.5,0) [color=gray,above=3mm,right=0mm,font=\fontsize{9}{10.2}] {semantic};
..   \node at (3.5,0) [color=gray,below=3mm,right=0mm,font=\fontsize{9}{10.2}] {analysis};
.. 
..   \node[draw=black, rectangle, fill=green!20, font=\fontsize{10}{10.2}] (ast2)  
..   at (7,0)  {Decorated AST};
.. 
..   \draw[->,very thick] (fst)  -- (ast1) ;
..   \draw[->,very thick] (ast1) -- (ast2) ;

Syntax
******

We use RedBaron_ to parse the *Python* code. For *headers*, *OpenMP* and *OpenAcc* we use textX_ 

.. _RedBaron: https://github.com/PyCQA/redbaron

.. _textX: https://github.com/igordejanovic/textX


In order to achieve *syntax analysis*, we first use *RedBaron* to get the **FST** (Full Syntax Tree), then we convert its nodes to our *sympy* **AST**. During this stage

- variables are described as *sympy* **Symbol** objects

.. note:: a **Symbol** can be viewed as a variable with **undefined type**

In the *semantic analysis* process, we *decorate* our *AST* and

- use **type inference** to get the type of every *symbol*

- change *Symbol*  objects to **Variable** when it is possible 


.. note:: since our target language is *Fortran*, we only convert variables that have a *type*. 

Full Syntax Tree (FST)
^^^^^^^^^^^^^^^^^^^^^^

Abstract Syntax Tree (AST)
^^^^^^^^^^^^^^^^^^^^^^^^^^


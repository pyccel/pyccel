Python
******

We use RedBaron_ to parse the *Python* code.

.. _RedBaron: https://github.com/PyCQA/redbaron


Refactoring Pyccel
^^^^^^^^^^^^^^^^^^

- use RedBaron_ instead of **textX** for parsing the *Python* code.

- keep **textX** for the *headers*, *OpenMP* and *OpenAcc*.

.. todo:: add the new diagram


.. tikz:: Constructing the AST for pure python code (no OpenMP/OpenACC). 

  \node[draw=black, rectangle, fill=red!40] (fst)  
  at (0,0)  {FST};

  \node at (0.5,0) [color=gray,above=3mm,right=0mm,font=\fontsize{10}{10.2}] {syntax};
  \node at (0.5,0) [color=gray,below=3mm,right=0mm,font=\fontsize{10}{10.2}] {analysis};

  \node[draw=black, rectangle, fill=red!20, font=\fontsize{10}{10.2}] (ast1)  
  at (3,0)  {AST};

  \node at (3.5,0) [color=gray,above=3mm,right=0mm,font=\fontsize{9}{10.2}] {semantic};
  \node at (3.5,0) [color=gray,below=3mm,right=0mm,font=\fontsize{9}{10.2}] {analysis};

  \node[draw=black, rectangle, fill=green!20, font=\fontsize{10}{10.2}] (ast2)  
  at (7,0)  {Decorated AST};

  \draw[->,very thick] (fst)  -- (ast1) ;
  \draw[->,very thick] (ast1) -- (ast2) ;

In order to achieve *syntax analysis*, we first use *RedBaron* to get the **FST** (Full Syntax Tree), then we convert its nodes to our *sympy* **AST**. During this stage

- variables are described as *sympy* **Symbol** objects

.. note:: a **Symbol** can be viewed as a variable with **undefined type**

In the *semantic analysis* process, we *decorate* our *AST* and

- use **type inference** to get the type of every *symbol*

- change *Symbol*  objects to **Variable** when it is possible 


.. note:: since our target language is *Fortran*, we only convert variables that have a *type*. 

Full Syntax Tree (FST)
^^^^^^^^^^^^^^^^^^^^^^

===================================   =============  =========  =========  
         RedBaron Nodes                  AST Nodes    phase 1    phase 2
===================================   =============  =========  =========
ArgumentGeneratorComprehensionNode
AssertNode                             Assert             +
AssignmentNode                         Assign             +
AssociativeParenthesisNode                                +
AtomtrailersNode                                          +
BinaryNode
BinaryOperatorNode                                        + 
BooleanOperatorNode                                       +
CallNode                                                  + 
CallArgumentNode                                          +
ClassNode
CommaNode
ComparisonNode                                            +
ComprehensionIfNode
ComprehensionLoopNode
DecoratorNode
DefArgumentNode                                           +
DelNode                                                   +
DictArgumentNode                                          +
DictNode                                                  +
DictComprehensionNode
DottedAsNameNode
DotNode                                                   +
ElifNode                                                  +
ElseNode                                                  +
EndlNode                                                  +
ExceptNode                                                x
ExecNode
FinallyNode                                               x
ForNode                                                   +
FromImportNode
FuncdefNode
GeneratorComprehensionNode
GetitemNode
GlobalNode
IfNode                                                    +
IfelseblockNode                                           +
ImportNode
IntNode                                                   +
LambdaNode
ListArgumentNode
ListComprehensionNode
ListNode                                                  +
NameAsNameNode
PrintNode                                                 +
RaiseNode                                                 x
ReprNode
ReturnNode                                                +
SetNode
SetComprehensionNode
SliceNode
SpaceNode
StringChainNode
TernaryOperatorNode
TryNode                                                   x
TupleNode                                                 +
UnitaryOperatorNode                                       +
YieldNode                                                 x
YieldAtomNode                                             x
WhileNode                                                 +
WithContextItemNode
WithNode
===================================   =============  =========  =========  

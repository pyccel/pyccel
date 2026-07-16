# pylint: disable=protected-access
"""
Implementations of methods that override existing methods in the SemanticParser.
"""

from pyccel.ast.core import CodeBlock, Comment


def _visit_CodeBlock(self, block):
    visited_block = super(type(self), self)._visit_CodeBlock(block)
    lines = visited_block.body
    new_lines = []
    for l in lines:
        ast = l.python_ast
        if ast:
            new_lines.append(Comment(f"{self._filename} : Line {ast.lineno}"))
        new_lines.append(l)

    new_block = CodeBlock(new_lines)
    visited_block.invalidate_node()
    return new_block

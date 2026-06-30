

def _visit_CodeBlock(self, block):
    visited_block = self._visit(block)
    lines = visited_block.body
    new_lines = []
    for l in lines:
        ast = l.python_ast
        if ast:
            new_lines.append(Comment("{self._filename} : Line {ast.line}"))
        new_lines.append(l)

    new_block = CodeBlock(visited_block)
    visited_block.invalidate_node()
    return new_block

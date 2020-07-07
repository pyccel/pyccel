"""Extended AST with CommentLine nodes
======================================

"""

from numpy import array, logical_and, where
from ast   import AST, If as IfNode, parse

class CommentLine(AST):
    """"New AST node representing a comment line"""

    _fields = ("s",)

    def __init__(self, s, lineno, col_offset):
        super().__init__()
        self.s = s
        self.lineno     = lineno
        self.col_offset = col_offset

def get_comments(code): 
    lines = code.split("\n")
    comments        = [] 
    lineno_comments = []
    else_no         = []

    for index, line in enumerate(lines):
        c = line.strip()
        if c.startswith("#"): 
            lineno_comments.append(index + 1) 
            comments.append(CommentLine(c, index+1, line.index('#')))
        elif c.startswith('else'):
            if c[4:].strip().startswith(':'):
                else_no.append(index + 1)
 
    return array(lineno_comments), array(comments), array(else_no)

def extend_tree(code):
    comment_lines_no, comments, else_no = get_comments(code)
    tree = parse(code)
    if len(tree.body) == 0:
        if len(comments) > 0:
            tree.body        = [comments[0]]
            comment_lines_no = comment_lines_no[1:]
            comments         = comments[1:]

    insert_comments(tree, comment_lines_no, comments, else_no)
    return tree

def get_last_lineno(ast):
    while hasattr(ast, 'body'):
        if ast.orelse:
            ast = ast.orelse[-1]
        else:
            ast = ast.body[-1]
    return ast.lineno

def insert_comments(ast, comment_lines_no, comments, else_no, attr='body', col_offset = None):
    if len(comments) == 0:
        return
    body        = getattr(ast, attr)

    if attr == 'orelse':
        assert col_offset is not None
    else:
        col_offset = body[0].col_offset

    node_lineno = body[0].lineno
    ind         = 0

    # insert in the beginning of block
    indices = where(node_lineno>comment_lines_no)[0]

    if len(indices)>0:
        ind              = indices[-1] + 1
        body             = comments[:ind].tolist() + body
        comment_lines_no = comment_lines_no[ind:]
        comments         = comments[ind:]

    # insert between two stmts
    while ind<len(body)-1 and len(comments)>0 :

        ind              = ind + 1
        previous_stmt    = body[ind-1]
        next_stmt        = body[ind]
        next_node_lineno = next_stmt.lineno
        first_comment_lineno = comment_lines_no[0]

        if next_node_lineno<first_comment_lineno:
            continue

        if not hasattr(previous_stmt, 'body'):
            for k, comment_line_no_k in enumerate(comment_lines_no):
                if next_node_lineno<comment_line_no_k:
                    body             = body[:ind]+ comments[:k].tolist() + body[ind:]
                    comment_lines_no = comment_lines_no[k:]
                    comments         = comments[k:]
                    ind             += k
                    break
        else:
            orelse = hasattr(previous_stmt, 'orelse') and previous_stmt.orelse

            if orelse:
                previous_stmt_body_last_lineno  = previous_stmt.orelse[0].lineno
                elif_orelse = len(previous_stmt.orelse) == 1 and isinstance(previous_stmt.orelse[0], IfNode)
            else:
                previous_stmt_body_last_lineno  = get_last_lineno(previous_stmt.body[-1])

            k = -1
            for k, comment_line_no_k in enumerate(comment_lines_no):
                if previous_stmt_body_last_lineno<comment_line_no_k:
                    if orelse and elif_orelse:
                        break
                    elif col_offset >= comments[k].col_offset and not orelse:
                        break
            else:
                k = k+1

            if orelse and not elif_orelse and k>0 :
                expr = logical_and(logical_and(else_no >= comment_lines_no[0], else_no <= comment_lines_no[k-1]), else_no<=previous_stmt_body_last_lineno)
                if expr.any():
                    k = where(else_no[expr][-1]<=comment_lines_no[:k])[0][0]

            insert_comments(previous_stmt, comment_lines_no[:k], comments[:k], else_no)
            comment_lines_no = comment_lines_no[k:]
            comments         = comments[k:]

            if orelse:
                previous_stmt_orelse_last_lineno  = get_last_lineno(previous_stmt.orelse[-1])

                k = -1
                for k, comment_line_no_k in enumerate(comment_lines_no):
                    if previous_stmt_orelse_last_lineno<comment_line_no_k:
                        if col_offset >= comments[k].col_offset:
                            break
                else:
                    k = k+1

                insert_comments(previous_stmt, comment_lines_no[:k], comments[:k], else_no, 'orelse', col_offset)
                comment_lines_no = comment_lines_no[k:]
                comments         = comments[k:]

            for k, comment_line_no_k in enumerate(comment_lines_no):
                if next_node_lineno<comment_lines_no_k:
                    body             = body[:ind] + comments[:k].tolist() + body[ind:]
                    comment_lines_no = comment_lines_no[k:]
                    comments         = comments[k:]
                    ind             += k
                    break

    last_stmt = body[-1]
    if hasattr( last_stmt, 'body'):
        orelse            = hasattr(last_stmt, 'orelse') and last_stmt.orelse

        if orelse:
            body_last_lineno  = last_stmt.orelse[0].lineno
            elif_orelse = len(last_stmt.orelse) == 1 and isinstance(last_stmt.orelse[0], IfNode)
        else:
            body_last_lineno  = get_last_lineno(last_stmt.body[-1])

        k = -1
        for k, comment_line_no_k in enumerate(comment_lines_no):
            if body_last_lineno<comment_line_no_k:
                if orelse and elif_orelse:
                    break
                if col_offset >= comments[k].col_offset and not orelse:
                    break
        else:
            k = k+1 

        if orelse and not elif_orelse and k>0:
            expr = logical_and(logical_and(else_no >= comment_lines_no[0], else_no <= comment_lines_no[k-1]), else_no<=body_last_lineno)
            if expr.any():
                k = where(else_no[expr][-1]<=comment_lines_no[:k])[0][0]

        insert_comments(last_stmt, comment_lines_no[:k], comments[:k], else_no)

        comment_lines_no = comment_lines_no[k:]
        comments         = comments[k:]

        if orelse:
            orelse_last_lineno  = get_last_lineno(last_stmt.orelse[-1])
            k = -1
            for k, comment_line_no_k in enumerate(comment_lines_no):
                if orelse_last_lineno<comment_line_no_k:
                    if col_offset >= comments[k].col_offset:
                        break
            else:
                k = k+1

            insert_comments(last_stmt, comment_lines_no[:k], comments[:k], else_no, 'orelse', col_offset)
            comment_lines_no = comment_lines_no[k:]
            comments      = comments[k:]

    # insert in the rest of the comments in the end of a block
    setattr(ast, attr, body + comments.tolist())


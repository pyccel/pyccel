# coding: utf-8
import redbaron
from redbaron import RedBaron
from redbaron import IntNode

code = 'x = 1'
red  = RedBaron(code)
stmt = red[0]

#type(stmt)
#stmt.target
#type(stmt.target)
#stmt.value
#type(stmt.value)

from pyccel.ast import Variable
from pyccel.ast import Assign

def datatype_from_redbaron(node):
    """Returns the pyccel datatype of a RedBaron Node."""
    if isinstance(node, IntNode):
        return 'int'
    else:
        raise NotImplementedError('only int is treated')

dtype = datatype_from_redbaron(stmt.value)
var   = Variable(dtype, str(stmt.target))

expr = Assign(var, stmt.value)

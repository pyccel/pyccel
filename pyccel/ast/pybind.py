from .basic import PyccelAstNode

class FunctionDeclaration(PyccelAstNode):
    _my_attribute_nodes = ('_func', '_mod_var', '_orig_func')
    def __init__(self, func, mod_var, orig_func):
        self._func = func
        self._mod_var = mod_var
        self._orig_func = orig_func
        super().__init__()

    @property
    def mod_var(self):
        return self._mod_var

    @property
    def function(self):
        return self._func

    @property
    def orig_function(self):
        return self._orig_func

# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from .ccode import CCodePrinter, import_dict
from pyccel.ast.variable import DottedName
from pyccel.ast.core import Import

__all__ = ["CCudaCodePrinter", "cudaccode"]

cuda_library_headers = {
    "cuda_runtime",
    "stdint",
    # ...
}

class CCudaCodePrinter(CCodePrinter):
    """ Cuda C Code printer
    """
    def _print_ModuleHeader(self, expr):
        name = expr.module.name
        # TODO: Add classes and interfaces
        funcs = '\n\n'.join('{};'.format(self.function_signature(f)) for f in expr.module.funcs)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *map(Import, self._additional_imports)]
        imports = '\n'.join(self._print(i) for i in imports)

        return ('#ifndef {name}_H\n'
                '#define {name}_H\n\n'
                '{imports}\n\n'
                '#ifndef ___cplusplus\nextern "C" {{\n#endif\n'
                #'{classes}\n\n'
                '{funcs}\n\n'
                '#ifndef ___cplusplus\n}}\n#endif\n'
                #'{interfaces}\n\n'
                '#endif // {name}_H\n').format(
                        name    = name.upper(),
                        imports = imports,
                        funcs   = funcs)


def cudaccode():
    pass

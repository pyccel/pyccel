

class CMakeGen:
    def __init__(self):
        pass

    def build_CompileTarget(self, expr):
        kernel_lib = '{expr.name}_kernel'
        args = ',\n    '.join([kernel_lib, 'OBJECT', expr.file])
        lib_cmd = 'add_library({args})\n'

        # TODO: Target link libraries

        wrap_args = ',\n    '.join([f"'{expr.name}'", 'MODULE', 'WITH_SOABI', 
                                    *expr.wrapper_files])
        wrap_cmd = f'Python_add_library({wrap_args})\n'

        cmds = [lib_cmd, wrap_cmd]

        if expr.is_exe:
            args = ',\n    '.join(['prog_{expr.name}', expr.program_file, 'OUTPUT_NAME', f"'{expr.name}'"])
            cmds.append(f'add_executable({args})\n')

            args = ',\n    '.join(['prog_{expr.name}', 'PUBLIC', kernel_lib, 'cwrapper'])
            cmds.append(f'target_link_libraries({args})')

        return '\n\n'.join(cmds)

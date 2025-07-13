

class MesonGen:
    def __init__(self):
        pass

    def build_CompileTarget(self, expr):
        kernel_lib = '{expr.name}_kernel'
        args = [f"'{kernel_lib}'", *expr.files]
        files = ',\n    '.join(args)
        lib_cmd = f'{kernel_lib} = library({args})\n'

        wrap_args = ',\n    '.join([f"'{expr.name}'", *expr.wrapper_files,
                                    f'dependencies : [cwrapper_dep]',
                                    f'link_with: {kernel_lib})'])
        wrap_cmd = f'py.extension_module({wrap_args})\n'

        cmds = [lib_cmd, wrap_cmd]

        if expr.is_exe:
            args = [expr.program_file, f'link_with: {kernel_lib})'])
            cmds.append(f'prog_{expr.name} = executable({args})\n')

        return '\n\n'.join(cmds)


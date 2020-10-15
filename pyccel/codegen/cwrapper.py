
def print_list(l):
    if isinstance(l,str):
        l = [l]
    return '[{0}]'.format(', '.join("r'{0}'".format(li) for li in l))

def create_c_setup(mod_name,
        dependencies,
        compiler,
        include = '',
        libs = '',
        libdirs = '',
        flags = ''):
    code  = "from setuptools import Extension, setup\n\n"

    deps  = ", ".join('r"{0}.c"'.format(d) for d in dependencies)

    mod = '"{mod}"'.format(mod=mod_name)

    files = "[{0}]".format(deps)

    include_str = ('include_dirs = {0}'.format(print_list(include))
                   if include else None)

    libs_str    = ('libraries = {0}'.format(print_list(libs))
                   if libs else None)
    libdirs_str = ('library_dirs = {0}'.format(print_list(libdirs))
                   if libdirs else None)

    flags_str   = ('extra_compile_args = {0}'.format(print_list(flags.strip().split()))
                   if flags else None)

    args = [mod, files, include_str, libs_str, libdirs_str, flags_str]
    args = ', '.join(a for a in args if a is not None)

    code += "extension_mod = Extension({args})\n\n".format(args=args)
    code += "setup(name = \"" + mod_name + "\", ext_modules=[extension_mod])"
    return code


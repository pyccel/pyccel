from itertools import chain
import os
import shutil
import subprocess
import sys

from pyccel.codegen.compiling.project import DirTarget
from pyccel.codegen.compiling.library_config import recognised_libs, ExternalLibInstaller

from .build_gen import BuildSystemHandler

class MesonHandler(BuildSystemHandler):

    def _generate_CompileTarget(self, expr):
        obj_lib = f'{expr.name}_objs'
        dep_name = f'{expr.name}_dep'
        mod_name = f"'{expr.pyfile.stem}'"

        out_folder = f"'{expr.pyfile.parent}'"

        lib_args = [mod_name, f"'{expr.file.name}'", 'build_by_default: false']

        dep_args = ["include_directories: ['.']"]

        deps = {*(f"{t.name}_dep" for t in expr.dependencies),
                  *(f"{r}_dep" for r in recognised_libs \
                    if any(d == r or d.startswith(f"{r}/") \
                    for d in expr.stdlib_dependencies))}
        if deps:
            dep_args.append(f"dependencies: [{', '.join(deps)}]")

        lib_args += dep_args
        args = ',\n  '.join(lib_args)
        lib_cmd = f'{obj_lib} = static_library({args})\n'

        args = ',\n  '.join(chain([f'objects: {obj_lib}.extract_all_objects(recursive: true)'], dep_args))
        dep_cmd = f'{dep_name} = declare_dependency(\n  {args})'


        args = ',\n  '.join([mod_name,
                             *(f"'{w.name}'" for w in expr.wrapper_files),
                             f"dependencies: [{dep_name}, cwrapper_dep]",
                              "install: true",
                             f"install_dir: {out_folder}"])
        wrap_cmd = f'py.extension_module({args})\n'

        cmds = [lib_cmd, dep_cmd, wrap_cmd]

        if expr.is_exe:
            args = ',\n  '.join((mod_name, f"'{expr.program_file.name}'",
                                 f"dependencies: [{dep_name}]",
                                  "install: true",
                                 f"install_dir: {out_folder}"))
            cmds.append(f'prog_{expr.name} = executable({args})\n')

        return '\n\n'.join(cmds)

    def _generate_DirTarget(self, expr):
        targets = []
        for t in expr.targets:
            if isinstance(t, DirTarget):
                code, subdir_cmd = self._generate_DirTarget(t)
                filename = self._pyccel_dir / t.folder.relative_to(self._root_dir) / 'meson.build'

                if self._verbose > 1:
                    print(">>> Printing :: ", filename)

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(code)
                targets.append(subdir_cmd)
            else:
                targets.append(self._generate_CompileTarget(t))

        code = '\n'.join(targets)

        return code, f"subdir('{expr.folder.stem}')\n"

    def generate(self, expr):
        languages = ', '.join(f"'{l}'" for l in expr.languages)
        project_decl = f"project('{expr.project_name}', {languages}, meson_version: '>=1.1.0')\n"

        # Python dependencies
        py_import = f"py = import('python').find_installation('{sys.executable}')\n"
        numpy_dep = "numpy_dep = dependency('numpy')\n"
        py_deps = ''.join(("# Python dependencies\n", py_import, numpy_dep))

        sections = [project_decl, py_deps]

        for d in expr.stdlib_deps:
            lib_install = recognised_libs.get(d, None)
            if isinstance(lib_install, ExternalLibInstaller):
                sections.append(f"{d}_dep = dependency('{lib_install.name}')\n")
            else:
                sections.append(f"subdir('{d}')\n")

        target_code, _ = self._generate_DirTarget(expr._dir_info)

        sections.append(target_code)

        code = '\n'.join(sections)

        filename = self._pyccel_dir / 'meson.build'
        if self._verbose > 1:
            print(">>> Printing :: ", filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code)

    def compile(self):
        capture_output = (self._verbose == 0)
        meson = shutil.which('meson')
        buildtype = 'debug' if self._debug_mode else 'release'

        if self._verbose:
            print(">> Running meson")
        subprocess.run([meson, 'setup', 'build', '--buildtype', buildtype], check=True,
                       cwd=self._pyccel_dir, capture_output=capture_output, env = os.environ)
        subprocess.run([meson, 'compile', '-C', 'build'], check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output, env = os.environ)
        subprocess.run([meson, 'install', '-C', 'build'], check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output, env = os.environ)

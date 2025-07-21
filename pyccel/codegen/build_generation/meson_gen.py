import shutil
import subprocess
import sys
from .build_gen import BuildSystemHandler
from pyccel.codegen.compiling.project import DirTarget

class MesonHandler(BuildSystemHandler):

    def _generate_CompileTarget(self, expr):
        kernel_dep = f'{expr.name}_dep'
        mod_name = f"'{expr.pyfile.stem}'"

        out_folder = f"'{expr.pyfile.parent}'"

        args = [mod_name, f"'{expr.file.name}'"]

        to_link = {f"{t.name}_dep" for t in expr.dependencies}
        to_link.update(expr.stdlib_dependencies)
        if to_link:
            link_args = ', '.join(to_link)
            args.append(f"link_with: [{link_args}]")

        files = ',\n    '.join(args)
        lib_cmd = f'{kernel_dep} = library({files})\n'

        wrap_args = [mod_name,
                     *(f"'{w.name}'" for w in expr.wrapper_files),
                     f'dependencies : [cwrapper_dep]',
                     f'link_with: {kernel_dep}',
                     "install: true",
                     f"install_dir: {out_folder}"]
        extra_inc_dirs = [f"{t.name}_inc_dir" for t in expr.dependencies if t.file.suffix == '.f90']
        if extra_inc_dirs:
            extra_inc_dirs_code = ', '.join(extra_inc_dirs)
            wrap_args.insert(-2, f'include_directories: [{extra_inc_dirs_code}]')
        wrap_args_code = ',\n    '.join(wrap_args)
        wrap_cmd = f'py.extension_module({wrap_args_code})\n'

        cmds = [lib_cmd, wrap_cmd]

        if expr.file.suffix == '.f90':
            wrap_incdir = f"{expr.name}_inc_dir = include_directories('.')\n"
            cmds.append(wrap_incdir)

        if expr.is_exe:
            link_args = ', '.join([kernel_dep, *to_link])
            args = ',\n    '.join((mod_name, f"'{expr.program_file.name}'",
                                   f"link_with: [{link_args}]", "install: true",
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
        project_decl = f"project('{expr.project_name}', {languages})\n"

        # Python dependencies
        py_import = f"py = import('python').find_installation('{sys.executable}')\n"
        numpy_dep = "numpy_dep = dependency('numpy')\n"
        py_deps = ''.join(("# Python dependencies\n", py_import, numpy_dep))

        sections = [project_decl, py_deps]

        for d, folder in expr.stdlib_deps.items():
            sections.append(f"subdir('{folder}')\n")

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

        subprocess.run([meson, 'setup', 'build', '--buildtype', buildtype], check=True,
                       cwd=self._pyccel_dir, capture_output=capture_output)
        subprocess.run([meson, 'compile', '-C', 'build'], check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output)
        subprocess.run([meson, 'install', '-C', 'build'], check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output)

from pathlib import Path
import shutil
import subprocess
import sys
from .build_gen import BuildSystemHandler
from pyccel.codegen.compiling.project import DirTarget

class CMakeHandler(BuildSystemHandler):

    def _generate_CompileTarget(self, expr):
        kernel_target = f'{expr.name}_kernel'
        mod_name = expr.pyfile.stem

        out_folder = expr.pyfile.parent

        args = '\n    '.join([kernel_target, 'STATIC', expr.file.name])
        cmds = [f'add_library({args})\n']

        to_link = [f"{t.name}_kernel" for t in expr.dependencies]
        if to_link:
            link_args = '\n    '.join([kernel_target, 'PUBLIC', *to_link])
            cmds.append(f"target_link_libraries({link_args})\n")

        if expr.file.suffix == '.c':
            args = '\n    '.join([kernel_target, 'PUBLIC', '${CMAKE_CURRENT_SOURCE_DIR}'])
            cmds.append(f'target_include_directories({args})\n')
        elif expr.file.suffix == '.f90':
            args = '\n    '.join([kernel_target, 'PUBLIC', '${CMAKE_CURRENT_BINARY_DIR}'])
            cmds.append(f'target_include_directories({args})\n')

        wrap_args = '\n    '.join([f'{kernel_target}_so', 'MODULE', 'WITH_SOABI',
                                    *[w.name for w in expr.wrapper_files]])
        cmds.append(f'Python_add_library({wrap_args})\n')

        target_args = '\n    '.join([f'{kernel_target}_so', 'PROPERTIES', 'OUTPUT_NAME', mod_name])
        cmds.append(f'set_target_properties({target_args})')

        to_link.append('cwrapper')
        link_args = '\n    '.join([f'{kernel_target}_so', 'PUBLIC', *to_link, 'cwrapper'])
        cmds.append(f"target_link_libraries({link_args})\n")

        args = '\n    '.join(['TARGETS', f'{kernel_target}_so', 'DESTINATION', str(out_folder)])
        cmds.append(f"install({args})\n")

        if expr.is_exe:
            prog_target = f'prog_{expr.name}'
            args = '\n    '.join([prog_target, expr.program_file.name])
            cmds.append(f'add_executable({args})\n')

            args = '\n    '.join([prog_target, 'PROPERTIES', 'OUTPUT_NAME', mod_name])
            cmds.append(f'set_target_properties({args})\n')

            args = '\n    '.join([prog_target, 'PUBLIC', kernel_target])
            cmds.append(f'target_link_libraries({args})')

            args = '\n    '.join(['TARGETS', prog_target, 'DESTINATION', str(out_folder)])
            cmds.append(f"install({args})\n")

        return '\n\n'.join(cmds)

    def _generate_DirTarget(self, expr):
        targets = []
        for t in expr.targets:
            if isinstance(t, DirTarget):
                code, subdir_cmd = self._generate_DirTarget(t)
                filename = self._pyccel_dir / t.folder.relative_to(self._root_dir) / 'CMakeLists.txt'

                if self._verbose > 1:
                    print(">>> Printing :: ", filename)

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(code)
                targets.append(subdir_cmd)
            else:
                targets.append(self._generate_CompileTarget(t))

        code = '\n'.join(targets)

        return code, f"add_subdirectory({expr.folder.stem})\n"

    def generate(self, expr):
        cmake_min = 'cmake_minimum_required(VERSION 3.20)'

        languages = ' '.join(['LANGUAGES', *[l.capitalize() for l in expr.languages]])
        project_decl = f"project('{expr.project_name}' {languages})\n"

        pic_on = 'set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n'

        # Python dependencies
        version = sys.version_info

        py_import = (f'set(Python_ROOT_DIR {Path(sys.executable).parent.parent})\n'
                     f"find_package(Python {version.major}.{version.minor}.{version.micro} EXACT REQUIRED COMPONENTS Development NumPy)\n")

        sections = [cmake_min, project_decl, pic_on, py_import]

        for d, folder in expr.stdlib_deps.items():
            sections.append(f"add_subdirectory({folder})\n")

        target_code, _ = self._generate_DirTarget(expr._dir_info)

        sections.append(target_code)

        code = '\n'.join(sections)

        filename = self._pyccel_dir / 'CMakeLists.txt'
        if self._verbose > 1:
            print(">>> Printing :: ", filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code)

    def compile(self):
        capture_output = (self._verbose == 0)
        cmake = shutil.which('cmake')
        buildtype = 'Debug' if self._debug_mode else 'Release'

        subprocess.run([cmake, '-B', 'build', f'-DCMAKE_BUILD_TYPE={buildtype}'], check=True,
                       cwd=self._pyccel_dir, capture_output=capture_output)
        subprocess.run([cmake, '--build', 'build'], check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output)
        subprocess.run([cmake, '--install', 'build'], check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output)

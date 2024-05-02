# coding: utf-8

import os
import subprocess

# ...
def clean(filename):
    """
    removes the generated files: .pyccel and .f90

    filename: str
        name of the file to parse.
    """
    name = filename.split('.py')[0]
    for ext in ["f90", "pyccel"]:
        f_name = name + "." + ext
        cmd = "rm -f " + f_name
        os.system(cmd)
# ...

# ...
def make_tmp_file(filename, output_dir=None):
    """
    returns a temporary file of extension .pyccel that will be decorated with
    indent/dedent so that textX can find the blocks easily.

    filename: str
        name of the file to parse.

    output_dir: str
        directory to store pyccel file
    """
    name = filename.split('.py')[0]
    if output_dir:
        name = os.path.basename(name)
        name = os.path.join(output_dir, name)
    return name + ".pyccel"
# ...

# ...
def preprocess_as_str(lines):
    """
    The input python file will be decorated with
    indent/dedent so that textX can find the blocks easily.
    This function will write the output code in filename_out.

    lines: str or list
        python code as a string
    """
    if type(lines) == str:
        ls = lines.split("\n")
        lines = []
        for l in ls:
            lines.append(l + "\n")

#    # to be sure that we dedent at the end
#    lines += "\n"

    lines_new = ""

    def delta(line):
        l = line.lstrip(' ')
        n = len(line) - len(l)
        return n

    tab   = 4
    depth = 0
    old_line = ""
    annotated = ""
    for i,line in enumerate(lines):
        n = delta(line)
        is_empty   = (len(line.lstrip()) == 0)
        is_comment = False
        if not is_empty:
            is_comment = (line.lstrip()[0] == '#')
        skipped = is_empty or is_comment
        is_annotated = (line.lstrip()[0:2] == '#$')

        if n == depth * tab + tab:
            depth += 1
            lines_new += "indent" + "\n"
            lines_new += line
        elif not skipped:
            d = n // tab
            if (d > 0) or (n==0):
                old = delta(old_line)
                m = (old - n) // tab
                depth -= m
                for j in range(0, m):
                    lines_new += "dedent" + "\n"
                lines_new += annotated
                annotated = ""
            lines_new += line
        elif is_annotated:
            annotated += line
        else:
            lines_new += line
        if not skipped:
            old_line = line
    for i in range(0, depth):
        lines_new += "dedent" + "\n"
    lines_new += annotated

    return lines_new
# ...

# ...
def preprocess(filename, filename_out):
    """
    The input python file will be decorated with
    indent/dedent so that textX can find the blocks easily.
    This function will write the output code in filename_out.

    filename: str
        name of the file to parse.

    filename_out: str
        name of the temporary file that will be parsed by textX.
    """
    f = open(filename)
    lines = f.readlines()
    f.close()

    lines_new = preprocess_as_str(lines)

    f = open(filename_out, "w")
    for line in lines_new:
        f.write(line)
    f.close()
# ...

# ...
# TODO improve, when using mpi
def execute_file(binary):
    """
    Execute a binary file.

    binary: str
        the name of the binary file
    """
    cmd = binary
    if not ('/' in binary):
        cmd = "./" + binary
    os.system(cmd)
# ...

# ...
def get_extension(language):
    """
    returns the extension of a given language.

    language: str
        low-level target language used in the conversion
    """
    if language == "fortran":
        return "f90"
    else:
        raise ValueError("Only fortran is available")
# ...

# ...
def separator(n=40):
    """
    Creates a separator string.
    This is used to improve the readability of the generated code.

    n: int
        length of the separator
    """
    txt = "."*n
    comment = '!'
    return '{0} {1}\n'.format(comment, txt)
# ...

class Compiler(object):
    """Base class for Code compiler for the Pyccel Grammar"""
    def __init__(self, codegen, compiler,
                 flags=None,
                 accelerator=None,
                 binary=None,
                 debug=False,
                 inline=False,
                 include=[],
                 libdir=[],
                 libs=[],
                 ignored_modules=[]):
        """
        Constructor of the code compiler.

        codegen: pyccel.codegen.Codegen
            a generation code object.
        compiler: str
            used compiler for the target language.
        flags: str
            compiler flags
        accelerator: str
            name of the selected accelerator.
            One among ('openmp', 'openacc')
        binary: str
            name of the binary file to generate.
        debug: bool
            add some useful prints that may help for debugging.
        inline: bool
            set to True, if the file is being load inside a python session.
        include: list
            list of include directories paths
        libdir: list
            list of lib directories paths
        libs: list
            list of libraries to link with
        ignored_modules: list
            list of modules to ignore.
        """
        self._codegen         = codegen
        self._compiler        = compiler
        self._binary          = binary
        self._debug           = debug
        self._inline          = inline
        self._accelerator     = accelerator
        self._include         = include
        self._libdir          = libdir
        self._libs            = libs
        self._ignored_modules = ignored_modules

        if flags:
            self._flags = flags
        else:
            self._flags = self.construct_flags()

    @property
    def codegen(self):
        """Returns the used codegen"""
        return self._codegen

    @property
    def compiler(self):
        """Returns the used compiler"""
        return self._compiler

    @property
    def flags(self):
        """Returns the used flags"""
        return self._flags

    @property
    def binary(self):
        """Returns the used binary"""
        return self._binary

    @property
    def debug(self):
        """Returns True if in debug mode"""
        return self._debug

    @property
    def inline(self):
        """Returns True if in inline mode"""
        return self._inline

    @property
    def accelerator(self):
        """Returns the used accelerator"""
        return self._accelerator

    @property
    def include(self):
        """Returns include paths"""
        return self._include

    @property
    def libdir(self):
        """Returns lib paths"""
        return self._libdir

    @property
    def libs(self):
        """Returns libraries to link with"""
        return self._libs

    @property
    def ignored_modules(self):
        """Returns ignored modules"""
        return self._ignored_modules

    def construct_flags(self):
        """
        Constructs compiling flags
        """
        # TODO use constructor and a dict to map flags w.r.t the compiler
        _avail_compilers = ['gfortran', 'mpif90', 'pgfortran']

        compiler    = self.compiler
        debug       = self.debug
        inline      = self.inline
        accelerator = self.accelerator
        include     = self.include
        libdir      = self.libdir

        if not(compiler in _avail_compilers):
            raise ValueError("Only {0} are available.".format(_avail_compilers))

        flags = " -O3 "
        if compiler == "gfortran":
            if debug:
                flags += " -fbounds-check "

        if compiler == "mpif90":
            if debug:
                flags += " -fbounds-check "

        if not (accelerator is None):
            if accelerator == "openmp":
                flags += " -fopenmp "
            elif accelerator == "openacc":
                flags += " -ta=multicore -Minfo=accel "
            else:
                raise ValueError("Only openmp and openacc are available")

        if isinstance(include, str):
            include = [include]
        if len(include) > 0:
            flags += ' '.join(' -I{0}'.format(i) for i in include)

        if isinstance(libdir, str):
            libdir = [libdir]
        if len(libdir) > 0:
            flags += ' '.join(' -L{0}'.format(i) for i in libdir)

        return flags

    def compile(self, verbose=False):
        """
        Compiles the generated file.

        verbose: bool
            talk more
        """
        compiler  = self.compiler
        flags     = self.flags
        inline    = self.inline
        libs      = self.libs
        filename  = self.codegen.filename_out
        is_module = self.codegen.is_module
        modules   = self.codegen.modules

#        print('> ignored = ', self.ignored_modules)
#        print('> modules = ', modules)
        modules = [m for m in modules if not(m in self.ignored_modules)]

        binary = ""
        if self.binary is None:
            if not is_module:
                binary = filename.split('.')[0]
        else:
            binary = self.binary

        o_code = ''
        if not inline:
            if not is_module:
                o_code = "-o"
            else:
                flags += ' -c '
        else:
            o_code = '-m'
            flags  = '--quiet -c'
            binary = self.codegen.name
            compiler = 'f2py'

        m_code = ' '.join('{}.o '.format(m) for m in modules)

        if isinstance(libs, str):
            libs = libs.split(',')
            if len(libs) == 1:
                libs = libs[0].split(' ')
        if len(libs) > 0:
            libs = ' '.join(' -l{0}'.format(i) for i in libs)
        else:
            libs = ''

        cmd = '{0} {1} {2} {3} {4} {5} {6}'.format( \
            compiler, flags, m_code, filename, o_code, binary, libs)

        if verbose:
            print(cmd)

        output = subprocess.check_output(cmd, shell=True)

        if verbose:
            print(output)

        # write and save a log file in .pyccel/'filename'.log
        # ...
        def mkdir_p(dir):
            # type: (unicode) -> None
            if os.path.isdir(dir):
                return
            os.makedirs(dir)

        if inline:
            tmp_dir = '.pyccel'
            mkdir_p(tmp_dir)
            logfile = '{0}.log'.format(binary)
            logfile = os.path.join(tmp_dir, logfile)
            f = open(logfile, 'w')
            f.write(output)
            f.close()
        # ...

        self._binary = binary
# ...

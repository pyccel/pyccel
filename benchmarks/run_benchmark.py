from collections import namedtuple
import os
import re
import shutil
import subprocess

TestInfo = namedtuple('TestInfo', 'basename imports setup call')

verbose = False

test_cases = ['python', 'pyccel']
#test_cases += ['pypy']
#test_cases += ['pyccel_intel']
#test_cases += ['pyccel_c']
#test_cases += ['pyccel_intel_c']
test_cases += ['pythran']
test_cases += ['numba']
pypy           = False
pyccel_intel   = False
pyccel_c       = False
pyccel_intel_c = False
pythran        = True
numba          = True

pyperf = True
time_compliation = True

tests = [
    TestInfo('ackermann_mod.py',
        ['ackermann'],
        'import sys; sys.setrecursionlimit(3000)',
        'ackermann(3,8)'),
    TestInfo('bellman_ford_mod.py',
        ['bellman_ford_test'],
        '',
        'bellman_ford_test()'),
    TestInfo('dijkstra.py',
        ['dijkstra_distance_test'],
        '',
        'dijkstra_distance_test()'),
    TestInfo('md_mod.py',
        ['test_md'],
        '',
        'test_md(p_num = 1000, step_num = 500)'),
    TestInfo('ode.py',
        ['euler_humps_test', 'humps_fun'],
        'import numpy as np; tspan = np.array([0.,2.]); y0 = np.array([humps_fun(0.0)]);',
        'euler_humps_test(tspan, y0, 10000)'),
    TestInfo('ode.py',
        ['midpoint_explicit_humps_test', 'humps_fun'],
        'import numpy as np; tspan = np.array([0.,2.]); y0 = np.array([humps_fun(0.0)]);',
        'midpoint_explicit_humps_test(tspan, y0, 10000)'),
    TestInfo('ode.py',
        ['midpoint_fixed_humps_test', 'humps_fun'],
        'import numpy as np; tspan = np.array([0.,2.]); y0 = np.array([humps_fun(0.0)]);',
        'midpoint_fixed_humps_test(tspan, y0, 10000)'),
    TestInfo('ode.py',
        ['rk4_humps_test', 'humps_fun'],
        'import numpy as np; tspan = np.array([0.,2.]); y0 = np.array([humps_fun(0.0)]);',
        'rk4_humps_test(tspan, y0, 10000)'),
    TestInfo('cfd_python.py',
        ['linearconv_1d'],
        'import numpy as np; nx = 501; nt = 1500; nu = 0.3; dx = 2 / (nx-1);
        grid = np.linspace(0,2,nx);
        u0 = np.ones(nx);
        u0[int(.5 / dx):int(1 / dx + 1)] = 2;
        u = u0.copy();
        un = np.ones(nx)',
        'linearconv_1d(u, un, nt, nx, dt, dx, c)'),
    TestInfo('cfd_python.py',
        ['nonlinearconv_1d'],
        'import numpy as np; nx = 2001; c=1.; dt=0.00035; dx = 2 / (nx-1);
        grid = np.linspace(0,2,nx);
        u0 = np.ones(nx);
        u0[int(.5 / dx):int(1 / dx + 1)] = 2;
        u = u0.copy();
        un = np.ones(nx)',
        'nonlinearconv_1d(u, un, nt, nx, dt, dx)'),
    TestInfo('cfd_python.py',
        ['poisson_2d'],
        'import numpy as np; nx = 150; ny = 150; nt  = 100; dx = 2 / (nx-1);
        grid = np.linspace(0,2,nx);
        u0 = np.ones(nx);
        u0[int(.5 / dx):int(1 / dx + 1)] = 2;
        u = u0.copy();
        un = np.ones(nx)',
        'nonlinearconv_1d(u, un, nt, nx, dt, dx)')
]

timeit_cmd = ['pyperf', 'timeit', '--copy-env'] if pyperf else ['timeit']

flags = '-O3 -march=native -mtune=native -mavx -ffast-math'

accelerator_commands = {
        'pyccel'         : ['pyccel', '--language=fortran', '--flags='+flags],
        'pyccel_c'       : ['pyccel', '--language=c', '--flags='+flags],
        'pyccel_intel'   : ['pyccel', '--language=fortran', '--compiler=intel', '--flags='+flags],
        'pyccel_intel_c' : ['pyccel', '--language=c', '--compiler=intel', '--flags='+flags],
        'pythran'        : ['pythran']+flags.split()
        }


for t in tests:
    basename = t.basename

    test_file = os.path.join('code',basename)
    testname  = os.path.splitext(basename)[0]

    numba_basename = 'numba_'+basename
    numba_testname = 'numba_'+testname
    numba_test_file = os.path.join(os.path.dirname(test_file), numba_basename)

    new_folder = os.path.join('tmp',t.imports[0])

    os.makedirs(new_folder, exist_ok=True)
    shutil.copyfile(test_file, os.path.join(new_folder, basename))
    shutil.copyfile(numba_test_file, os.path.join(new_folder, numba_basename))
    os.chdir(new_folder)

    import_funcs = ', '.join(t.imports)
    exec_cmd  = t.call

    for case in test_cases:
        setup_cmd = 'from {testname} import {funcs};'.format(
                testname = numba_testname if case == 'numba' else testname,
                funcs = import_funcs)
        setup_cmd += t.setup
        print("-------------------")
        print("   ",case)
        print("-------------------")
        run_test = True
        if case in accelerator_commands:
            cmd = accelerator_commands[case].copy()+[basename]

            if time_compliation:
                cmd = ['time'] + cmd

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True)
            out, err = p.communicate()

            if p.returncode != 0:
                print("Compilation Error!")
                print(err)
                run_test = False
                continue

            if time_compliation:
                regexp = re.compile('([0-9.]*)user\s*([0-9.]*)system')
                r = regexp.findall(err)
                assert len(r) == 1
                times = [float(ri) for ri in r[0]]
                cpu_time = sum(times)
                print("Compilation CPU time : ", cpu_time)

        cmd = ['pypy'] if case=='pypy' else ['python3']
        cmd += ['-m'] + timeit_cmd + ['-s', setup_cmd, exec_cmd]

        print(cmd)

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True)
        out, err = p.communicate()

        if p.returncode != 0:
            print("Execution Error!")
            bench_str = '-'
            if verbose:
                print(err)
        else:
            if pyperf:
                regexp = re.compile('([0-9.]*) (..) \+- ([0-9.]*) (..)')
                r = regexp.search(out)
                assert r.group(2) == r.group(4)
                mean = r.group(1)
                stddev = r.group(3)
                units = r.group(2)

                bench_str = '{mean} $\pm$ {stddev}'.format(
                        mean=mean,
                        stdev=stddev)
            else:
                regexp = re.compile('([0-9]*) loops, best of ([0-9]*): ([0-9.]*) (\w*)')
                r = regexp.search(out)
                best = r.group(3)
                units = r.group(4)
                bench_str = best

        if case in accelerator_commands:
            p = subprocess.Popen(['pyccel-clean', '-s'])


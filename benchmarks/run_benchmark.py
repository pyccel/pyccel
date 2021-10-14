from collections import namedtuple
import os
import re
import shutil
import subprocess

TestInfo = namedtuple('TestInfo', 'name basename imports setup call')

verbose = False
latex_out = False

pyperf = False
time_compliation = True

test_cases = ['python']
#test_cases += ['pypy']
test_cases += ['pythran']
test_cases += ['numba']
test_cases += ['pyccel']
#test_cases += ['pyccel_intel']
#test_cases += ['pyccel_c']
#test_cases += ['pyccel_intel_c']
pypy           = False
pyccel_intel   = False
pyccel_c       = False
pyccel_intel_c = False
pythran        = True
numba          = True

tests = [
    TestInfo('Ackermann',
        'ackermann_mod.py',
        ['ackermann'],
        'import sys; sys.setrecursionlimit(3000)',
        'ackermann(3,8)'),
    TestInfo('Bellman Ford',
        'bellman_ford_mod.py',
        ['bellman_ford_test'],
        '',
        'bellman_ford_test()'),
    TestInfo('Dijkstra',
        'dijkstra.py',
        ['dijkstra_distance_test'],
        '',
        'dijkstra_distance_test()'),
    TestInfo('M-D',
        'md_mod.py',
        ['test_md'],
        '',
        'test_md(3, 1000, 500, 0.1)'),
    TestInfo('Euler',
        'ode.py',
        ['euler_humps_test', 'humps_fun'],
        'import numpy as np; tspan = np.array([0.,2.]); y0 = np.array([humps_fun(0.0)]);',
        'euler_humps_test(tspan, y0, 10000)'),
    TestInfo('Midpoint Explicit',
        'ode.py',
        ['midpoint_explicit_humps_test', 'humps_fun'],
        'import numpy as np; tspan = np.array([0.,2.]); y0 = np.array([humps_fun(0.0)]);',
        'midpoint_explicit_humps_test(tspan, y0, 10000)'),
    TestInfo('Midpoint Fixed',
        'ode.py',
        ['midpoint_fixed_humps_test', 'humps_fun'],
        'import numpy as np; tspan = np.array([0.,2.]); y0 = np.array([humps_fun(0.0)]);',
        'midpoint_fixed_humps_test(tspan, y0, 10000)'),
    TestInfo('RK4',
        'ode.py',
        ['rk4_humps_test', 'humps_fun'],
        'import numpy as np; tspan = np.array([0.,2.]); y0 = np.array([humps_fun(0.0)]);',
        'rk4_humps_test(tspan, y0, 10000)'),
    TestInfo('FD - L Convection',
        'cfd_python.py',
        ['linearconv_1d'],
        '''import numpy as np; nx=2001; nt=2000; c=1.; dt=0.0003;
        dx = 2 / (nx-1);
        grid = np.linspace(0,2,nx);
        u0 = np.ones(nx);
        u0[int(.5 / dx):int(1 / dx + 1)] = 2;
        u = u0.copy();
        un = np.ones(nx);''',
        'linearconv_1d(u, un, nt, nx, dt, dx, c)'),
    TestInfo('FD - NL Convection',
        'cfd_python.py',
        ['nonlinearconv_1d'],
        '''import numpy as np; nx = 2001; nt=2000; c=1.; dt=0.00035; dx = 2 / (nx-1);
        grid = np.linspace(0,2,nx);
        u0 = np.ones(nx);
        u0[int(.5 / dx):int(1 / dx + 1)] = 2;
        u = u0.copy();
        un = np.ones(nx)''',
        'nonlinearconv_1d(u, un, nt, nx, dt, dx)'),
    TestInfo('FD - Poisson',
        'cfd_python.py',
        ['poisson_2d'],
        '''import numpy as np; nx = 150; ny = 150; nt  = 100;
           xmin = 0; xmax = 2; ymin = 0; ymax = 1;
           dx = (xmax - xmin) / (nx - 1);
           dy = (ymax - ymin) / (ny - 1);
           p  = np.zeros((ny, nx));
           pd = np.zeros((ny, nx));
           b  = np.zeros((ny, nx));
           x  = np.linspace(xmin, xmax, nx);
           y  = np.linspace(xmin, xmax, ny);''',
        'poisson_2d(p, pd, b, nx, ny, nt, dx, dy)'),
    TestInfo('FD - Laplace',
        'cfd_python.py',
        ['laplace_2d'],
        '''import numpy as np; nx = 31; ny = 31; c = 1.; l1norm_target=1.e-4;
           dx = 2 / (nx - 1); dy = 2 / (ny - 1);
           p = np.zeros((ny, nx));
           x = np.linspace(0, 2, nx);
           y = np.linspace(0, 1, ny);
           p[:, 0] = 0; p[:, -1] = y;
           p[0, :] = p[1, :]; p[-1, :] = p[-2, :]''',
        'laplace_2d(p, y, dx, dy, l1norm_target)'),
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

cell_splitter = ' & ' if latex_out else ' | '
row_splitter = '\\\\\n\\hline\n' if latex_out else '\n'

test_cases_row = cell_splitter.join('{0: <25}'.format(s) for s in ['Code']+test_cases)
result_table = [test_cases_row]

possible_units = ['s','ms','us','ns']
latex_units = ['s','ms','\\textmu s','ns']

start_dir = os.getcwd()

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

    run_times = []
    run_units = []

    for case in test_cases:
        setup_cmd = 'from {testname} import {funcs};'.format(
                testname = numba_testname if case == 'numba' else testname,
                funcs = import_funcs)
        setup_cmd += t.setup.replace('\n','')
        print("-------------------")
        print("   ",case)
        print("-------------------")
        if case in accelerator_commands:
            cmd = accelerator_commands[case].copy()+[basename]

            if time_compliation:
                cmd = ['time'] + cmd

            print(cmd)

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True)
            out, err = p.communicate()

            if p.returncode != 0:
                print("Compilation Error!")
                print(err)
                run_times.append(None)
                run_units.append(None)
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
            run_times.append(None)
            run_units.append(None)
            print(err)
        else:
            print(out)
            if pyperf:
                regexp = re.compile('([0-9.]+) (\w\w?) \+- ([0-9.]+) (\w\w?)')
                r = regexp.search(out)
                assert r.group(2) == r.group(4)
                mean = float(r.group(1))
                stddev = float(r.group(3))
                units = r.group(2)

                bench_str = '{mean} $\pm$ {stddev}'.format(
                        mean=mean,
                        stddev=stddev)
                run_times.append((mean,stddev))
            else:
                regexp = re.compile('([0-9]+) loops?, best of ([0-9]+): ([0-9.]+) (\w*)')
                r = regexp.search(out)
                best = float(r.group(3))
                units = r.group(4)[:-2]
                bench_str = best
                run_times.append(best)

            run_units.append(possible_units.index(units))

        if case in accelerator_commands:
            p = subprocess.Popen(['pyccel-clean', '-s'])

    used_units = [u for u in run_units if u is not None]
    unit_index = round(sum(used_units)/len(used_units))
    units = latex_units[unit_index]
    row = [t.name + ' ('+units+')']

    mult_fact = [1000**(unit_index-u) if u is not None else None for u in run_units]

    print(run_times)

    if pyperf:
        for time,f in zip(run_times,mult_fact):
            if time is None:
                row.append('-')
            else:
                mean,stddev = time
                row.append('{mean} $\pm$ {stddev}'.format(
                            mean=mean*f,
                            stddev=stddev*f))
    else:
        for time,f in zip(run_times,mult_fact):
            if time is None:
                row.append('-')
            else:
                row.append(str(time*f))

    row = cell_splitter.join('{0: <25}'.format(s) for s in row)
    print(row)
    result_table.append(row)
    os.chdir(start_dir)

print(row_splitter.join(result_table))

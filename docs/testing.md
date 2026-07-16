# Testing Pyccel

Continuous testing runs on GitHub actions: <https://github.com/pyccel/pyccel/actions?query=branch%3Adevel>

To test your local Pyccel installation please use the command line tool `pyccel test` which runs all unit tests using pytest under the hood. Alternatively, if you want to have more fine-grained control over which tests you run (e.g. for debugging your local modifications to Pyccel), you can call Pytest directly with the following instructions.

We download the source code for a specific release, and with it the tests

```sh
curl -JLO "https://github.com/pyccel/pyccel/archive/refs/tags/v1.12.1.zip"
unzip pyccel-1.12.1.zip
cd pyccel-1.12.1/tests
```

or, in the case of `devel` branch on GitHub:

```sh
curl -JLO "https://github.com/pyccel/pyccel/archive/refs/heads/devel.zip"
unzip pyccel-devel.zip
cd pyccel-devel/tests
```

We start by running in parallel (with as many threads as possible) the single-process tests which do not create conflicts with other tests (this is very fast):

```sh
pytest -n auto -ra -m "not xdist_incompatible and c"
pytest -n auto -ra -m "not xdist_incompatible and not python and not c"
pytest -n auto -ra -m "not xdist_incompatible and python"
```

Next, we proceed with running the single-process tests which cannot run in parallel with other tests (this takes some time):

```sh
pytest -ra -m "xdist_incompatible"
```

Finally, we make sure that the `epyccel` command can be run from an MPI-parallel Python program:

```sh
mpirun -n 4 --oversubscribe pytest --with-mpi -ra epyccel/test_parallel_epyccel.py
```

import os
import shutil

__all__ = ['setup', 'teardown']

def setup():
    teardown()

def teardown(path_dir = None):
    if path_dir is None:
        path_dir = os.path.dirname(os.path.realpath(__file__))

    for root, _, files in os.walk(path_dir):
        for name in files:
            if name.startswith(".coverage"):
                shutil.copyfile(os.path.join(root,name),os.path.join(os.getcwd(),name))

    files = os.listdir(path_dir)
    for f in files:
        file_name = os.path.join(path_dir,f)
        if f == "__pyccel__" or f == "__epyccel__":
            shutil.rmtree( file_name )
        elif not os.path.isfile(file_name):
            teardown(file_name)
        elif not f.endswith(".py"):
            os.remove(file_name)

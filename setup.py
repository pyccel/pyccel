# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import setuptools
import sys
from setuptools.command.dist_info import dist_info
from setuptools.command.egg_info import egg_info

class PickleHeadersInstall(dist_info):
    """ Class to pickle headers in time for them to be collected
    by the MANIFEST.in treatment during the install command
    """
    def run(self):
        # Process .pyh headers and store their AST in .pyccel pickle files.
        from pyccel.commands.pyccel_init import pyccel_init

        pyccel_init()

        # Execute the classic dist_info command
        super().run()

class PickleHeadersWheel(egg_info):
    """ Class to pickle headers in time for them to be collected
    by the MANIFEST.in treatment while building a wheel
    """
    def run(self):
        # Process .pyh headers and store their AST in .pyccel pickle files.
        from pyccel.commands.pyccel_init import pyccel_init

        pyccel_init()

        # Execute the classic egg_info command
        super().run()

if __name__ == "__main__":
    setuptools.setup(cmdclass={
        "dist_info": PickleHeadersInstall,
        "egg_info": PickleHeadersWheel,
        })

# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import setuptools
from setuptools.command.egg_info import egg_info

class PickleHeaders(egg_info):
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
        "egg_info": PickleHeaders,
        })

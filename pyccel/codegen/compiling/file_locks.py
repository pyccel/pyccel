# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module handling classes which handle file locking to avoid deadlocks.
"""
from filelock import FileLock

class FileLockSet:
    """
    Class for grouping file locks.

    A class which groups file locks. By grouping these the locking can
    be handled via a context manager which reduces the risk of the locks
    not being correctly released.

    Parameters
    ----------
    locks : iterable[FileLock], optional
        The locks that should be stored in the FileLockSet.
    """
    def __init__(self, locks = ()):
        assert all(isinstance(l, FileLock) for l in locks)
        self._locks = list(locks)

    def __enter__(self):
        for l in self._locks:
            l.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        # Release the locks
        for l in reversed(self._locks):
            l.release()

    def append(self, new_lock):
        """
        Add a new lock to the FileLockSet.

        Add a new lock to the FileLockSet.

        Parameters
        ----------
        new_lock : FileLock
            The new lock.
        """
        assert isinstance(new_lock, FileLock)
        self._locks.append(new_lock)

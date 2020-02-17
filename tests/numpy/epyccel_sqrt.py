#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 08:29:32 2018

@author: calbert
"""
from numpy import sqrt
from pyccel import epyccel
from pyccel.decorators import types

@types('real')
def wurzel(a):
    return sqrt(a)

wurzel_fast = epyccel(wurzel)

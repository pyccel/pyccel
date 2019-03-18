#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 08:29:32 2018

@author: calbert
"""
from numpy import sqrt
from pyccel import epyccel

def wurzel(a):
    return sqrt(a)

header = '#$ header procedure wurzel(double) results(double)'
wurzel_fast = epyccel(wurzel,header)

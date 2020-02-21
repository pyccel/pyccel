#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 08:17:39 2018

@author: calbert
"""
import numpy as np

#$ header procedure static Ts(double, double, double, double, double) results(double)
def Ts(x, y, a, b, d):    
    if np.abs(x) < a/2.0 and np.abs(y) < b/2.0:
        r = 0.0
    if np.abs(x) < a/2.0 and b/2.0 < np.abs(y) < b/2.0+d:
        r = (np.abs(y)-b/2)/d
    if np.abs(y) < b/2.0 and a/2.0 < np.abs(x) < a/2.0+d:
        r = (np.abs(x)-a/2)/d

    dcorn2 = (np.abs(x)-a/2.0)**2+(abs(y)-b/2.0)**2

    if dcorn2 < d**2.0:
        r = np.sqrt(dcorn2)/d

    return r


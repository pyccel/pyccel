#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 08:17:39 2018

@author: calbert
"""
from numpy import abs, sqrt

#$ header procedure static Ts(double, double, double, double, double) results(double)
def Ts(x, y, a, b, d):    
    if abs(x) < a/2.0 and abs(y) < b/2.0:
        r = 0.0
    if abs(x) < a/2.0 and b/2.0 < abs(y) < b/2.0+d:
        r = (abs(y)-b/2)/d
    if abs(y) < b/2.0 and a/2.0 < abs(x) < a/2.0+d:
        r = (abs(x)-a/2)/d

    dcorn2 = (abs(x)-a/2.0)**2+(abs(y)-b/2.0)**2

    if dcorn2 < d**2.0:
        r = sqrt(dcorn2)/d

    return r


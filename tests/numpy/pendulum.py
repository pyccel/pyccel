#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:25:43 2018

@author: calbert
"""
#$ header dqdt(double)
#$ header dpdt(double)
#$ header verlet(double[:],double[:],double) results(double[:])

from numpy import sin

def dqdt(p): return 1.0*p
def dpdt(q): return -sin(q)

def verlet(q,p,dt):
    nt = len(q)
    pdot = dpdt(q[0])
    for kt in range(nt-1): # Verlet scheme
        p2 = p[kt] + 0.5*dt*pdot
        q[kt+1] = q[kt] + dt*dqdt(p2)
        pdot = dpdt(q[kt+1])
        p[kt+1] = p2 + 0.5*dt*pdot

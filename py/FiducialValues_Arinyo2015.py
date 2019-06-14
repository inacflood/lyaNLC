#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:31:07 2019

@author: iflood
"""
import numpy as np

def getFiducialValues(z):
    q1, q2, kvav, kp, av, bv = np.zeros(6)
    if z==2.2:
        q1, q2, kp, kvav, av, bv = [0.090, 0.316, 8.9, 0.493, 0.145, 1.54]
    elif z==2.4:
        q1, q2, kp, kvav, av, bv = [0.057, 0.368, 9.2, 0.480, 0.156, 1.57]
    elif z==2.6:
        q1, q2, kp, kvav, av, bv = [0.068, 0.390, 9.6, 0.483, 0.190, 1.61]
    elif z==2.8:
        q1, q2, kp, kvav, av, bv = [0.086, 0.417, 9.9, 0.493, 0.217, 1.63]
    elif z==3.0:
        q1, q2, kp, kvav, av, bv = [0.104, 0.444, 10.1, 0.516, 0.248, 1.66]
    else:
        print("Invalid z-value")
        return
    return q1, q2, kp, kvav, av, bv
    
        
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 21:09:33 2017

@author: lenna
"""


import numpy as np
import scipy.io as sio


x = np.zeros(5);
y = x+2

#saving a dictionary...
sio.savemat('prac_vec.mat',{'x':x,'y':y})

contents = sio.loadmat('prac_vec.mat')


contents['x']

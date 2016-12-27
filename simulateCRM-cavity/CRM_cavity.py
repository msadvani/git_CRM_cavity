# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:51:26 2016

@author: lenna
"""


from scipy.integrate import quad
import numpy as np
import random as rand;

K = .7; #carrying capacity mean
sig_K = .3;
m = .1;
sig_m = .5;
mu = .2;
sig = 1;
S = 100;
M = 200;



gamma = M/S;

def wfunc(j, d):
    def integrand(z, j, d):
        return np.exp(-z**2/2)*(z+d)**j 
    
    return (2*np.pi)**(-.5)*quad(integrand, -d, np.inf, args = (j,d))[0]
    
    
Delta_N = rand.random();
Delta_R = rand.random();
chi = rand.random();
sig_N = rand.random();
sig_R = rand.random();

maxCnt = 50;

for cnt in range(1, maxCnt):
    
    phi_N = wfunc(0, Delta_N);
    phi_R = wfunc(0, Delta_R);

    nu = -phi_N/(gamma*sig**2*chi);
    chi =  phi_R/(1 - sig**2*nu);

    avg_N = (sig_N/(gamma*sig**2*chi))*wfunc(1, Delta_N);
    avg_R = (sig_R/(1- sig**2*nu))*wfunc(1, Delta_R);


    q_N = (sig_N/(gamma*sig**2*chi))**2*wfunc(2, Delta_N);
    q_R = (sig_R/(1- sig**2*nu))**2*wfunc(2, Delta_R);

    sig_N_new =np.sqrt(sig**2*gamma*q_R + sig_m**2); 
    err = abs(sig_N_new - sig_N)
    #print(err)
    sig_N = sig_N_new;
    sig_R = np.sqrt(sig**2*q_N + sig_K**2);

    Delta_N = (mu*gamma*avg_R - m)/sig_N;
    Delta_R = (K  - mu*avg_N)/sig_R;
    

print(phi_N)
print(phi_R)
#print(nu)
#print(chi)
print(avg_N)
print(avg_R)
print(q_N)
print(q_R)
#print(Delta_N)
#print(Delta_R)





if err > 10**(-4):
   print('did not converge')
   print(err)
   
   
   
   
   
err1 = phi_N - wfunc(0,Delta_N)
err2 = phi_R - wfunc(0,Delta_R)
err3 = avg_N - (sig_N/(gamma*sig**2*chi))*wfunc(1, Delta_N)
err4 = avg_R - (sig_R/(1 - sig**2*nu))*wfunc(1, Delta_R)
err5 = q_N - (sig_N/(gamma*sig**2*chi))**2*wfunc(2,Delta_N)
err6 = q_R - (sig_R/(1 - sig**2*nu))**2*wfunc(2, Delta_R)
err7 = sig_R**2 -sig_K**2-sig**2*q_N
err8 = sig_N**2 - sig_m**2 - sig**2*gamma*q_R
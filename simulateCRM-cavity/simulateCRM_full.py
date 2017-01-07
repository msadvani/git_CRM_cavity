import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.integrate import quad

import scipy.io as sio


def Draw_random_parameters(S, M, sigma, mu, mu_K, sigma_K, mu_m, sigma_m):
    #This function draws c, m, K for simulating an ecosystem
    #We draw C uniformally from 0 to 1 and then scale to have appropriate mean and std

    #c=np.random.rand(S*M).reshape(S,M)
    #c=(c-np.mean(c))/np.std(c)*sigma/np.sqrt(S)+mu/S
    
    c = np.random.normal(mu/S, sigma/np.sqrt(S),S*M).reshape(S,M)
    
    #c=(c-np.mean(c))/np.std(c)*sigma/S+mu/S

    #Draw random carrying capactities
    K=np.random.normal(mu_K,sigma_K,M)
    
    #Draw minimum coefficient
    m=np.random.normal(mu_m, sigma_m, S)
        
    return [c,K,m]


def Get_vector_field_CRM(Y,t,par):
    #This function calculates vector field setting all amounts below epsilon to zero
    
    
    #unpack parameters
    [c,K,m,epsilon]=par
    M=len(K)
    S=len(m)
    
    
    
    #unpack resource abundances and species abundance
    R=Y[0:M]
    R[np.where(R<epsilon)]=0;
    N=Y[M:]
    N[np.where(N<epsilon)]=0;

    
    #Cacluate vector fields
    dN=  (c.dot(R)-m)*N
    dR= (K-R-N.dot(c))*R
    output_vector=np.concatenate((dR,dN))
    
    return output_vector
    
    
def Get_Jacobian_CRM(Y,t,par):
    
    #unpack parameters
    [c,K,m,epsilon]=par
    M=len(K)
    S=len(m)
    
    
    
    #unpack resource abundances and species abundance
    R=Y[0:M]
    R[np.where(R<epsilon)]=0;
    N=Y[M:]
    N[np.where(N<epsilon)]=0;    
    
    Jac=np.zeros((M+S,M+S))
    
 
    #Calculate dR_beta/dR_\alpha
    
    Jac[0:M,0:M]=np.diag(K-2*R-N.dot(c))
     
    #Calculate dR_beta/dN_j
    
    Jac[0:M,M:M+S]=np.einsum('ia,a->ai',c,R)
    
    #Calculate dN_i/dR_beta
    Jac[M:M+S,0:M]=np.einsum('ib, i->ib',c,N)
    
    #Calculate dN_i/dN_j
    
    Jac[M:M+S,M:M+S]=np.diag(c.dot(R)-m)

    return Jac

    
def wfunc(j, d):
    def integrand(z, j, d):
        return np.exp(-z**2/2)*(z+d)**j 
    
    return (2*np.pi)**(-.5)*quad(integrand, -d, np.inf, args = (j,d))[0]


def solveCavity(K, sig_K, m, sig_m, mu, sig, gamma):
    Delta_N = np.random.random();
    Delta_R = np.random.random();
    chi = np.random.random();
    sig_N = np.random.random();
    sig_R = np.random.random();
    
    maxCnt = 50;
    
    numPar = 10;
    param_vec= np.zeros(numPar); #keep track of the parameter values
    param_vec_next= np.zeros(numPar); #next parameter values    
    error_vec= np.zeros(numPar); #update displacement in each parameter
    err_RS_vec = np.zeros(8);
    for cnt in range(1, maxCnt):
        
        phi_N = wfunc(0, Delta_N);
        #print(phi_N - param_vec_next[0])
        phi_R = wfunc(0, Delta_R);
    
        nu = -phi_N/(gamma*sig**2*chi);
        chi =  phi_R/(1 - sig**2*nu);
    
        avg_N = (sig_N/(gamma*sig**2*chi))*wfunc(1, Delta_N);
        
        avg_R = (sig_R/(1- sig**2*nu))*wfunc(1, Delta_R);
    
    
        q_N = (sig_N/(gamma*sig**2*chi))**2*wfunc(2, Delta_N);
        q_R = (sig_R/(1- sig**2*nu))**2*wfunc(2, Delta_R);
    
        sig_N =np.sqrt(sig**2*gamma*q_R + sig_m**2); 
        sig_R = np.sqrt(sig**2*q_N + sig_K**2);
    
        Delta_N = (mu*gamma*avg_R - m)/sig_N;
        Delta_R = (K  - mu*avg_N)/sig_R;
    
        
        param_vec_next[:] = [phi_N, phi_R, avg_N, avg_R, q_N, q_R, nu, chi, Delta_N, Delta_R];
        #print(param_vec_next)
        #print(param_vec)


        error_vec = abs(param_vec_next - param_vec)
        
        err = sum(error_vec)
        param_vec[:] = [phi_N, phi_R, avg_N, avg_R, q_N, q_R, nu, chi, Delta_N, Delta_R];

        
    
        
        err_RS_vec[0] = phi_N - wfunc(0,Delta_N)
        err_RS_vec[1] = phi_R - wfunc(0,Delta_R)
        err_RS_vec[2] = avg_N - (sig_N/(gamma*sig**2*chi))*wfunc(1, Delta_N)
        err_RS_vec[3] = avg_R - (sig_R/(1 - sig**2*nu))*wfunc(1, Delta_R)
        err_RS_vec[4] = q_N - (sig_N/(gamma*sig**2*chi))**2*wfunc(2,Delta_N)
        err_RS_vec[5] = q_R - (sig_R/(1 - sig**2*nu))**2*wfunc(2, Delta_R)
        err_RS_vec[6] = sig_R**2 -sig_K**2-sig**2*q_N
        err_RS_vec[7] = sig_N**2 - sig_m**2 - sig**2*gamma*q_R
        
        err_RS = sum(np.abs(err_RS_vec));

        #print(err_RS) 

    print('convergence error:') 
    print(err)
    
    print('RS error:') 
    print(err_RS)
    
    
    return [phi_N, phi_R, avg_N, avg_R, q_N, q_R, nu, chi, Delta_N, Delta_R, err, err_RS]
    
    #print(phi_N)
    #print(phi_R)
    #print(nu)
    #print(chi)
    #print(avg_N)
    #print(avg_R)
    #print(q_N)
    #print(q_R)
    
      
    



#M=200.;
M = 50.;
S = 50.;
numVary = 5 #15

gamma = M/S;

sigmaSet=np.linspace(.1,8,numVary);

mu=.7;
mu_K=.9;
mu_m=.85;
sigma_K=.5;
sigma_m=.9;


#Cutoff on accuracy
epsilon=10**-4
numInit = 10;
numPar = 12;
cav_parSet = np.zeros((numPar, numVary, numInit))

numTrials_LV = 5; #20


dataLV_order_params = np.zeros((6,numTrials_LV,numVary));
N_dataLV = np.zeros((S, numTrials_LV,numVary));
R_dataLV = np.zeros((M, numTrials_LV,numVary));



for gcnt in range(0, numVary):
    sigma = sigmaSet[gcnt];
    
    for icnt in range(0, numInit):
        cav_par = solveCavity(mu_K, sigma_K, mu_m, sigma_m, mu, sigma, gamma)
        #print(cav_par)
    
        for pcnt in range(0,numPar):
            cav_parSet[pcnt,gcnt,icnt] = cav_par[pcnt]
    
        
    #LV simulations
    for cnt in range(0, numTrials_LV):
        print([cnt+1,numTrials_LV])
        
        
        #Create coefficients and append cuttoff
        par=Draw_random_parameters(S, M, sigma, mu, mu_K, sigma_K, mu_m, sigma_m)
        par.append(epsilon)
    
        R_ini=np.random.rand(M);
        N_ini=np.random.rand(S);
        
        t0=0;
        t1=10**4
        
        Y_ini=np.concatenate((R_ini,N_ini))
               
        t=np.linspace(t0,t1, num=1000)#
        Y= odeint(Get_vector_field_CRM,Y_ini,t,args=(par,),Dfun=Get_Jacobian_CRM,atol=10**-3)
        
        # Calculate the various cavity quantities
        R=Y[-1,0:M]
        N=Y[-1,M:M+S]
        
        N[N<10**-3]=0
        R[R<10**-3]=0
        
        N0=N[np.nonzero(N)]
        R0=R[np.nonzero(R)]
        
        S_star=N0.size
        M_star=R0.size
        
        S_star=S_star/1.
        M_star=M_star/1.
        
        phi_N=S_star/S
        phi_R=M_star/M
        
        N_avg=np.mean(N)
        R_avg=np.mean(R)
        
        q_N=np.mean(N**2)
        q_R=np.mean(R**2)
    
        N_dataLV[:,cnt,gcnt] = N;
        R_dataLV[:,cnt,gcnt] = R;
    

        dataLV_order_params[0,cnt,gcnt] = phi_N
        dataLV_order_params[1,cnt,gcnt] = phi_R
        dataLV_order_params[2,cnt,gcnt] = N_avg
        dataLV_order_params[3,cnt,gcnt] = R_avg
        dataLV_order_params[4,cnt,gcnt] = q_N
        dataLV_order_params[5,cnt,gcnt] = q_R

    
max_cav_err = .01;


cav_parSet_converged = copy.copy(cav_parSet);  #removing non-converging cases
#np.zeros(np.size(cav_parSet))

cav_parSet_converged[0:9,cav_parSet[10,:,:]>max_cav_err] = ['nan'];
    
titleSet = ['phi_S', 'phi_M', 'avg_N', 'avg_R', 'q_N', 'q_R']
numFig = np.size(titleSet);
#for fcnt in range(0, numFig):
#    plt.figure(fcnt)
#    plt.title(titleSet[fcnt])
#    for icnt in range(0, numInit):    
#        plt.plot(sigmaSet, cav_parSet[fcnt,:,icnt], 'ro')
    


        
        
for fcnt in range(0, numFig):
    plt.figure(fcnt)
    plt.title(titleSet[fcnt])
    for icnt in range(0, numInit):    
        plt.plot(sigmaSet, cav_parSet_converged[fcnt,:,icnt], 'ro')
        
    for cnt in range(0, numTrials_LV):
        plt.plot(sigmaSet, dataLV_order_params[fcnt,cnt,:], 'b+')
        
    
for fcnt in range(numFig, 2*numFig):
    rcnt = fcnt-numFig;
    plt.figure(fcnt)
    plt.title(titleSet[rcnt])
    for icnt in range(0, numInit):    
        plt.plot(sigmaSet, cav_parSet[rcnt,:,icnt], 'ro')    

        
  
        
paramData = [mu_K, sigma_K, mu_m, sigma_m, mu,gamma]
        


sio.savemat('LV_data2.mat',{'sigmaSet':sigmaSet,'cav_parSet_converged':cav_parSet_converged, 'paramData':paramData, 'cav_parSet':cav_parSet, 'dataLV_order_params':dataLV_order_params, 'N_dataLV':N_dataLV, 'R_dataLV':R_dataLV,'M':M,'S':S})

LV_data = sio.loadmat('LV_data2.mat')


LV_data['cav_parSet_converged']
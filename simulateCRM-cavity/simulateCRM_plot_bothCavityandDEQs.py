import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.integrate import quad


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
        
    print('convergence error:') 
    print(err)
    
    return [phi_N, phi_R, avg_N, avg_R, q_N, q_R, nu, chi, Delta_N, Delta_R]
    
    #print(phi_N)
    #print(phi_R)
    #print(nu)
    #print(chi)
    #print(avg_N)
    #print(avg_R)
    #\print(q_N)
    #print(q_R)
    
      
    



#M=200.;
M = 20.;
S = 20.;
numVary = 2

gamma = M/S;

sigmaSet=np.linspace(.1,4,numVary);
mu=1.;
mu_K=1.;
mu_m=1.;
sigma_K=1.;
sigma_m=1.;


#Cutoff on accuracy
epsilon=10**-4
numInit = 10;
numPar = 10;
cav_parSet = np.zeros((numPar, numVary, numInit))


numTrials_LV = 40;
numInit_LV = 10; #see how much difference initialization makes. What sort of variety of solutions is there... 

dataSet = np.zeros((6,numTrials_LV,numInit_LV,numVary));


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
    
        for icnt in range(0, numInit_LV):
            #Set initial condition
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
        
        
        
        
            dataSet[0,cnt,icnt,gcnt] = phi_N
            dataSet[1,cnt,icnt,gcnt] = phi_R
            dataSet[2,cnt,icnt,gcnt] = N_avg
            dataSet[3,cnt,icnt,gcnt] = R_avg
        
            dataSet[4,cnt,icnt,gcnt] = q_N
            dataSet[5,cnt,icnt,gcnt] = q_R


    



plt.figure(1)
plt.title('phi_S')

for icnt in range(0, numInit):
    plt.plot(sigmaSet, cav_parSet[0,:,icnt], 'ro')
    
for icnt in range(0, numInit_LV):
    for cnt in range(0, numTrials_LV):
        plt.plot(sigmaSet, dataSet[0,cnt,icnt,:], 'b+')
    
plt.xlabel('sigma')
plt.show()


plt.figure(2)
plt.title('phi_M')

for icnt in range(0, numInit):
    plt.plot(sigmaSet, cav_parSet[1,:,icnt], 'ro')
    
for icnt in range(0, numInit_LV):
    for cnt in range(0, numTrials_LV):
        plt.plot(sigmaSet, dataSet[1,cnt,icnt,:], 'b+')    

plt.xlabel('sigma')
plt.show()


plt.figure(3)
plt.title('avg_N')
for icnt in range(0, numInit):
    plt.plot(sigmaSet, cav_parSet[2,:,icnt], 'ro')
    
    
for icnt in range(0, numInit_LV):
    for cnt in range(0, numTrials_LV):
        plt.plot(sigmaSet, dataSet[2,cnt,icnt,:], 'b+')
        
plt.axis([0,max(sigmaSet),-.01,10.0])
plt.xlabel('sigma')
plt.show()



plt.figure(4)
plt.title('avg_R')
for icnt in range(0, numInit):
    plt.plot(sigmaSet, cav_parSet[3,:,icnt], 'ro')
    
for icnt in range(0, numInit_LV):
    for cnt in range(0, numTrials_LV):
        plt.plot(sigmaSet, dataSet[3,cnt,icnt,:], 'b+')
        
plt.xlabel('sigma')
plt.show()





plt.figure(5)
plt.title('q_N')
for icnt in range(0, numInit):
    plt.plot(sigmaSet, cav_parSet[4,:,icnt], 'ro')
for icnt in range(0, numInit_LV):
    for cnt in range(0, numTrials_LV):
        plt.plot(sigmaSet, dataSet[4,cnt,icnt,:], 'b+')
    
plt.xlabel('sigma')
plt.show()

plt.figure(6)
plt.title('q_R')
for icnt in range(0, numInit):
    plt.plot(sigmaSet, cav_parSet[5,:,icnt], 'ro')
    
for icnt in range(0, numInit_LV):
    for cnt in range(0, numTrials_LV):
        plt.plot(sigmaSet, dataSet[5,cnt,icnt,:], 'b+')
plt.xlabel('sigma')
plt.show()




plt.figure(7)
plt.title('nu')
for icnt in range(0, numInit):
    plt.plot(sigmaSet, cav_parSet[6,:,icnt], 'ro')
plt.xlabel('sigma')
plt.show()

plt.figure(8)
plt.title('chi')
for icnt in range(0, numInit):
    plt.plot(sigmaSet, cav_parSet[7,:,icnt], 'ro')
plt.xlabel('sigma')
plt.show()



plt.figure(9)
plt.title('Delta_N')
for icnt in range(0, numInit):
    plt.plot(sigmaSet, cav_parSet[8,:,icnt], 'ro')
plt.xlabel('sigma')
plt.show()

plt.figure(10)
plt.title('Delta_R')
for icnt in range(0, numInit):
    plt.plot(sigmaSet, cav_parSet[9,:,icnt], 'ro')
plt.xlabel('sigma')
plt.show()








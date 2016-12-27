
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import ode




def Draw_random_parameters(S, M, sigma, mu, mu_K, sigma_K, mu_m, sigma_m):
    #This function draws c, m, K for simulating an ecosystem
    #We draw C uniformally from 0 to 1 and then scale to have appropriate mean and std

    c=np.random.rand(S*M).reshape(S,M)
    c=(c-np.mean(c))/np.std(c)*sigma/S+mu/S


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



S=5;
M=10;
    
sigma=1.;
mu=0.2;
mu_K=0.7;
mu_m=0.1
sigma_K=0.3;
sigma_m=0.5;

#Cutoff on accuracy
epsilon=10**-4


#Create coefficients and append cuttoff
par=Draw_random_parameters(S, M, sigma, mu, mu_K, sigma_K, mu_m, sigma_m)
par.append(epsilon)

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

N=N[np.nonzero(N)]
R=R[np.nonzero(R)]

S_star=N.size
M_star=R.size

S_star=S_star/1.
M_star=M_star/1.

phi_N=S_star/S
phi_R=M_star/M

N_avg=np.mean(N)
R_avg=np.mean(R)

q_N=np.mean(N**2)
q_R=np.mean(R**2)


#Plot figure
plt.figure(1)
plt.semilogy(t,Y[:,M:M+S])
plt.show()


plt.figure(2)
plt.semilogy(t,Y[:,0:M])
plt.show()
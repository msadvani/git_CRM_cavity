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
    
    return [phi_N, phi_R, avg_N, avg_R, q_N, q_R, nu, chi]
      
      
    


numTrials=50
    
S=100;
M=200;
gamma = M/S;
    
sigma=1.;
mu=0.2;
mu_K=0.7;
mu_m=0.1
sigma_K=0.3;
sigma_m=0.5;



cav_par = solveCavity(mu_K, sigma_K, mu_m, sigma_m, mu, sigma, gamma)
print(cav_par)

#Cutoff on accuracy
epsilon=10**-4


numInit = 3;
dataSet = np.zeros((6, numTrials,numInit))

for cnt in range(0, numTrials-1):
    print([cnt+1,numTrials])
    
    
    #Create coefficients and append cuttoff
    par=Draw_random_parameters(S, M, sigma, mu, mu_K, sigma_K, mu_m, sigma_m)
    par.append(epsilon)
    
    
    for icnt in range(0, numInit-1):
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
    
    
    
    
        dataSet[0,cnt,icnt] = phi_N
        dataSet[1,cnt,icnt] = phi_R
        dataSet[2,cnt,icnt] = N_avg
        dataSet[3,cnt,icnt] = R_avg
    
        dataSet[4,cnt,icnt] = q_N
        dataSet[5,cnt,icnt] = q_R

#phi_N_avg = np.mean(dataSet[0,:])
#q_N_med = np.median(dataSet[1,:])
#print(phi_N_avg)
#print(q_N_med)


#from tempfile import TemporaryFile
#outfile = TemporaryFile()
#np.save(outfile, cav_par, dataSet)
#outfile.seek(0) # Only needed here to simulate closing & reopening file

#>>> np.load(outfile)

plt.figure(1)

plt.title('phi_N, phi_R')
plt.plot([0,1], [cav_par[0],cav_par[1]], 'ro')
#plt.errorbar(x, y, xerr=0.2, yerr=0.4)
plt.errorbar([0,1], [np.mean(dataSet[0,:]),np.mean(dataSet[1,:])], [2.*np.sqrt(1/numTrials)*np.std(dataSet[0,:]),2.*np.sqrt(1/numTrials)*np.std(dataSet[1,:])],fmt = 'o')
#plt.plot([0,1], [np.mean(dataSet[0,:]),np.mean(dataSet[1,:])], 'r+')


plt.axis([-1, 2, 0, 1.5])

plt.show()

plt.figure(2)

plt.title('<N>, <R>')

plt.plot([2,3], [cav_par[2],cav_par[3]], 'ro')

#plt.plot([2,3], [np.median(dataSet[2,:]),np.median(dataSet[3,:])], 'r+')
plt.errorbar([2,3], [np.median(dataSet[2,:]),np.median(dataSet[3,:])], [2.*np.sqrt(1/numTrials)*np.std(dataSet[2,:]),2.*np.sqrt(1/numTrials)*np.std(dataSet[3,:])],fmt = 'o')



plt.axis([1, 4, 0, 1.5])

plt.show()



plt.figure(3)


plt.title('q_N, q_R')

plt.plot([4,5], [cav_par[4],cav_par[5]], 'ro')

plt.errorbar([4,5], [np.median(dataSet[4,:]),np.median(dataSet[5,:])], [2.*np.sqrt(1/numTrials)*np.std(dataSet[4,:]),2.*np.sqrt(1/numTrials)*np.std(dataSet[5,:])],fmt = 'o')

#plt.plot([4,5], [np.median(dataSet[4,:]),np.median(dataSet[5,:])], 'r+')


plt.axis([3, 6, 0, 1.5])

plt.show()




#plt.plot([1,2,3,4], [1,4,9,16], 'ro')



#Plot figure
#plt.figure(1)
#plt.semilogy(t,Y[:,M:M+S])
#plt.show()
#
#
#plt.figure(2)
#plt.semilogy(t,Y[:,0:M])
#plt.show()
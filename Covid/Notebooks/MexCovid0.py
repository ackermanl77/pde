# coding=utf-8 
from __future__ import division
from scipy import integrate
#from xlrd import open_workbook
import numpy as np
#import scipy as sp
import pylab as pl
import pytwalk
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
#import emcee
#import time
import corner




model = 1


TotalNumIter = 50000
burnin       = 1999
LastNumIter  = 2000
NumParams    = 4
xi           = 2e-6


if model == 1: #SIR
    
    data=pd.read_csv('covidMexico.csv')
    mu          = 0.000046948
    nu          = 0.00001589
#    kappa = 0.00003 #p[3]    
    N     = 128932753
    
    windowYlim_D_vs_S=3600
    windowYlim_I = 3600
    windowYlim_R = 550
    save_results_to = 'MODEL1/'
    
elif  model== 2: # SEIR
    
    data=pd.read_csv('covidMexico.csv')
    
    mu          = 0.000046948
    nu          = 0.00001589
#    kappa = 0.00003 #p[3]    
    N     = 128932753
    S0 = 0.999998459
    I0 = 8e-9
    A0 = 0.000001533
    windowYlim_D_vs_S=3600
    windowYlim_I = 3600
    windowYlim_R = 550
    save_results_to = 'MODEL2/'
   
    
    
elif  model == 3: # # SEsQRP
    
    data=pd.read_csv('covidMexico.csv')
    
    mu          = 0.000046948
    nu          = 0.00001589
#    kappa = 0.00003 #p[3]    
    N     = 128932753
    S0 = 0.999998459
    I0 = 8e-9
    A0 = 0.000001533
    windowYlim_D_vs_S=3600
    windowYlim_I = 3600
    windowYlim_R = 550
    save_results_to = 'MODEL3/'
      
       
elif  model == 4: # TRI-TO
    
    data=pd.read_csv('TRI-TO-HIV-AIDS.csv',header = None)
      
    
else :
    print('Invalid model number')

# # HUC
Infected = data["New cases (HUC)"]
Removed  = data["Deaths (HUC)"] 

# datahub
# Infected = data["New cases (datahub)"]
# Removed  = data["Deaths (datahub)"] 

## OWD
#Infected = data["New cases (OWD)"]
#Removed  = data["Deaths (OWD)"] + data["Recovered (noData)"]

# # DGE
# Infected = data["New cases (DGE)"]
# Removed  = data["Deaths (DGE)"] 

#ttime = np.linspace(0.0,float(len(Infected)),len(Infected))
#ttime = list(float(i) for i in range(len(Infected)))
ttime = np.array([float(i) for i in range( len(Infected) )])

fig0= pl.figure()
plt.plot(ttime, Infected,'r.', ttime, Removed,'b.')
#plt.show()
pl.savefig(save_results_to + 'NewCases_and_Deaths.eps')

# plt.plot(ttime, Removed)
# plt.show()

#Infected = data["New cases (DGE)"]
#Removed  = data["Deaths (DGE)"] + data["Recovered (DGE)"]
#ttime = np.linspace(0.0,float(len(Infected)),len(Infected))
#plt.plot(ttime, Infected, ttime, Removed)
#plt.show()

#Infected_HUC = data["New cases (HUC)"]
#Infected_DH = data["New cases (datahub)"]
#Infected_OWD = data["New cases (OWD)"]
#Infected_DGE = data["New cases (DGE)"]
#
#
#plt.plot(ttime, Infected_HUC, ttime, Infected_DH, 
#         ttime,Infected_OWD , ttime, Infected_DGE )
#plt.show()
#
#plt.plot(ttime, Infected_HUC, ttime, Infected_DGE ) 
#plt.show()

# Initial conditions
S0 = (N-Infected[0])/N
I0 = Infected[0]/N
R0 = 0

# pesos para la cuadratura trapezoidal
weigths = np.ones(11)
weigths[0] = 0.5
weigths[-1] = 0.5


def modelo(x,t,p):
    
  """
  p[0]: beta transmision rate
  p[1]: sigma recovery rate
  p[2]: K=1/rho
  p[3]: kappa death rate

  x[0]: Susceptibles
  x[1]: Infecteds
  x[2]: Deaths
  """  
  
  fx = np.zeros(3)
  
  fx[0] = mu - (p[0] * x[1] + nu + xi   )*x[0]
  fx[1] = (p[0] * x[1] + xi ) *x[0] - (p[1] + nu )*x[1]
  fx[2] =  p[3]*x[1]  
  return fx




def solve(p):
    x0 = np.array([S0,I0,R0])
    nn = len(ttime)
    dt = 1.0/(10.0*nn)
    n_quad = 10*nn+1
    t_quad = np.linspace(ttime[0],ttime[-1],n_quad)
    soln = integrate.odeint(modelo,x0,t_quad,args=(p,))
    result_I = np.zeros(nn)
    result_R = np.zeros(nn)
    
    for k in np.arange(len(Infected)):       
        x_s = soln[10*k:10*(k+1)+1,0]
        x_i = soln[10*k:10*(k+1)+1,1]
        incidence_I = (p[0]*x_i+xi)*x_s
        incidence_R = ( p[3])*x_i
        result_I[k] = dt*np.dot(weigths,incidence_I)
        result_R[k] = dt*np.dot(weigths,incidence_R)      
    return p[2]*result_I,result_R
    
    
def energy(p):
    if support(p):
        my_soln_I, my_soln_R = solve(p)
        ind1 = np.where(my_soln_I < 10**-8)
        ind2 = np.where(my_soln_R < 10**-8)
        my_soln_I[ind1] = np.ones(len(ind1))*10**-8
        my_soln_R[ind2] = np.ones(len(ind2))*10**-8
        log_likelihood1 = -np.sum(my_soln_I*N-Infected*np.log(my_soln_I*N)) 
        log_likelihood2 = -np.sum(my_soln_R*N-Removed*np.log(my_soln_R*N))
        #log_likelihood = -np.sum(np.linalg.norm(my_soln*N-flu_data))**2/10.0**2
#        print(log_likelihood)
       # gamma distribution parameters for p[0] = beta
        k1 = 1.0
        theta1 = 1.0
        # gamma distribution parameters for p[1] = sigma
        k2 = 1.0
        theta2 = 1.0
        # gamma distribution parameters for p[2] = K1
        k3 = 1.0
        theta3 = 10.0
        # gamma distribution parameters for p[2] = K2
        k4 = 1.0
        theta4 = 10.0
        a1 = (k1-1)*np.log(p[0])- (p[0]/theta1)
        a2 = (k2-1)*np.log(p[1])- (p[1]/theta2) 
        a3 = (k3-1)*np.log(p[2])- (p[2]/theta3)
        a4 = (k4-1)*np.log(p[3])- (p[3]/theta4)
 
#        log_prior = a1 + a2 + a3
        log_prior = a1 + a2 + a3 + a4
        return -log_likelihood1 -log_likelihood2  - log_prior
    return -np.infty








def support(p):
    rt = True
    rt &= (0.0 < p[0] < 1.0)
    rt &= (0.0 < p[1] < 1.0)
    rt &= (0.0 < p[2] < 20.0)
    rt &= (0.0 < p[3] < 1.0)
    return rt

# p - parámetros p = (beta, K)
def init():
    p = np.zeros(4)
    p[0] = np.random.uniform(low=0.0,high=1.0)
    p[1] = np.random.uniform(low=0.0,high=1.0)
    p[2] = np.random.uniform(low=0.0,high=20.0)
    p[3] = np.random.uniform(low=0.0,high=1.0)
    return p


def euclidean(v1, v2):
    return sum((q1-q2)**2 for q1, q2 in zip(v1, v2))**.5

if __name__=="__main__": 
#    nn = len(flu_ttime)
#    print(nn)
#    input("Press Enter to continue...") 
#    burnin = 5000
    sir = pytwalk.pytwalk(n=NumParams,U=energy,Supp=support)
    sir.Run(T=TotalNumIter,x0=init(),xp0=init())
     
    ppc_samples_I = np.zeros((LastNumIter,len(ttime)))
    ppc_samples_R = np.zeros((LastNumIter,len(ttime)))
    
   
    
    fig0= pl.figure()
    ax0 = pl.subplot(111)
    sir.Ana(start=burnin)
    pl.savefig(save_results_to + 'trace_plot.eps')

    fig2 = pl.figure()
    ax2 = pl.subplot(111)
    qq = sir.Output[sir.Output[:,-1].argsort()] # MAP
    my_soln_I,my_soln_R = solve(qq[0,:]) # solve for MAP
    ax2.plot(ttime,my_soln_I*N,'b')
    ax2.plot(ttime,my_soln_R*N,'g')
    
    for k in np.arange(LastNumIter): # last 1000 samples
        ppc_samples_I[k],ppc_samples_R[k]=solve(sir.Output[-k,:])
        sample_I, sample_R = solve(sir.Output[-k,:]) 
        ax2.plot(ttime,sample_I*N,"#888888", alpha=.25)
        ax2.plot(ttime,sample_R*N,"#888888", alpha=.25) 
        
    ax2.plot(ttime,Infected,'r.')
    ax2.plot(ttime,Removed,'b.')
    pl.ylim(0.0,windowYlim_D_vs_S)
    pl.savefig(save_results_to + 'data_vs_samples.eps')
    samples = sir.Output[burnin:,:-1]
    #samples[:,1] *= N
    #samples[:,2] *= N
    map = qq[0,:-1]
    #map[1] *= N
    #map[2] *= N    
    range = np.array([(0.85*x,1.15*x) for x in map])
    corner.corner(samples,labels=[r"$\beta$", r"$\sigma$" , r"$K1$",  r"$\kappa$"],truths=map,range=range)
    pl.savefig(save_results_to + 'corner.eps')
    
    

    mean_ppc_I = np.percentile(ppc_samples_I,q=50.0,axis=0)
    mean_ppc_R = np.percentile(ppc_samples_R,q=50.0,axis=0)
    
    CriL_ppc_I = np.percentile(ppc_samples_I,q=2.5,axis=0)
    CriU_ppc_I = np.percentile(ppc_samples_I,q=97.5,axis=0)
    
    CriL_ppc_R = np.percentile(ppc_samples_R,q=2.5,axis=0)
    CriU_ppc_R = np.percentile(ppc_samples_R,q=97.5,axis=0)
    
 #   print(np.shape(mean_ppc))
    print(np.shape(CriL_ppc_I))
    print(np.shape(CriU_ppc_I))
    
    print(np.shape(CriL_ppc_R))
    print(np.shape(CriU_ppc_R))
    
    
    mean_ppc_beta  = samples[:,0].mean(axis=0)
    mean_ppc_sigma = samples[:,1].mean(axis=0)
    mean_ppc_K     = samples[:,2].mean(axis=0)
    mean_ppc_kappa = samples[:,3].mean(axis=0)
    
    CriL_ppc_beta = np.percentile(samples[:,0],q=2.5,axis=0)
    CriU_ppc_beta = np.percentile(samples[:,0],q=97.5,axis=0)
    
    CriL_ppc_sigma = np.percentile(samples[:,1],q=2.5,axis=0)
    CriU_ppc_sigma = np.percentile(samples[:,1],q=97.5,axis=0)
    
    CriL_ppc_K    = np.percentile(samples[:,2],q=2.5,axis=0)
    CriU_ppc_K    = np.percentile(samples[:,2],q=97.5,axis=0)
    
    CriL_ppc_kappa = np.percentile(samples[:,3],q=2.5,axis=0)
    CriU_ppc_kappa = np.percentile(samples[:,3],q=97.5,axis=0)
    
 #   print(np.shape(mean_ppc))
 
    print(mean_ppc_beta)
    print(mean_ppc_sigma)
    print(mean_ppc_K)
    print(mean_ppc_kappa)
 
    print(CriL_ppc_beta)
    print(CriU_ppc_beta)
    
    print(CriL_ppc_sigma)
    print(CriU_ppc_sigma)
    
    print(CriL_ppc_K)
    print(CriU_ppc_K)
    
    print(CriL_ppc_kappa)
    print(CriU_ppc_kappa)
    
    
    
    
    
    pl.figure()
    ax2 = pl.subplot(111)
    ax2.plot(ttime,Infected,'r.')
#   my_soln_S,my_soln_I = solve(qq[0,:]) # solve for MAP
    ax2.plot(ttime,mean_ppc_I*N,'b',lw=2)
#    ax2.plot(ttime,mean_ppc_I,'b',lw=2)
    pl.ylim(0.0,windowYlim_I)
    ax2.fill_between(ttime, CriL_ppc_I*N, CriU_ppc_I*N, color='b',alpha=0.2)
    pl.savefig(save_results_to + 'BandsPredictionI.pdf')
    
    
    pl.figure()
    ax2 = pl.subplot(111)
    ax2.plot(ttime,Removed,'r.')
    pl.ylim(0.0,windowYlim_R)
#    ax2.plot(ttime,my_soln_I*N,'c',lw=2)
    ax2.plot(ttime,mean_ppc_R*N,'c',lw=2)
    ax2.fill_between(ttime, CriL_ppc_R*N, CriU_ppc_R*N, color='c',alpha=0.2)
    pl.savefig(save_results_to + 'BandsPredictionR.pdf')
    
    
    

    pl.figure()
    pl.hist(samples[:,0]/(samples[:,1]+nu),normed=True)
    pl.savefig(save_results_to + 'R_0.eps')
    
   
    pl.figure()
    alpha, beta= 1.0, 1.0
    data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
    myHist = pl.hist(data, 100, normed=True)
    pl.hist(samples[:,0],normed=True)
    pl.xlim(0.0,1.0)
    pl.ylim(0.0,10.0)
    pl.savefig(save_results_to + 'beta_prior_vs_posterior.eps')
    
    pl.figure()
    alpha, beta= 1.0, 1.0
    data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
    myHist = pl.hist(data, 100, normed=True)
    pl.hist(samples[:,1],normed=True)
    pl.xlim(0.0,1.0)
    pl.ylim(0.0,10.0)
    pl.savefig(save_results_to + 'sigma_prior_vs_posterior.eps')
    
    plt.figure()
    alpha, beta= 1.0, 1.0
    data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
    myHist = plt.hist(data, 100, normed=True)
    plt.hist(samples[:,2],normed=True)
    plt.savefig(save_results_to + 'K1_prior_vs_posterior.eps')
    
    plt.figure()
    alpha, beta= 1.0, 1.0
    data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
    myHist = plt.hist(data, 100, normed=True)
    plt.hist(samples[:,3],normed=True)
    plt.savefig(save_results_to + 'kappa_prior_vs_posterior.eps')
    
#    plt.figure()
#    alpha, beta= 1.0, 1.0
#    data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
#    myHist = plt.hist(data, 100, normed=True)
#    plt.hist(samples[:,3],normed=True)
#    plt.savefig(save_results_to + 'K2_prior_vs_posterior.eps')
    
    
    print('Norm Square of Infected data =')
    print(euclidean(sample_I*N, Infected))
    print('Norm Square of Deaths data =')
    print(euclidean(sample_R*N, Removed))














































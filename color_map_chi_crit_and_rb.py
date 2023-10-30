#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:56:05 2023

@author: s1984454
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
from RLoKi import RLoKi,determine_critical_chi
from RLoKi_asymptotics import RLoKi_asymp,determine_critical_chi_asymp
from LoKi import LoKi
import pickle 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

def calculate_chi_crits_rbs(Psi,n_mu,n_eps):
    epsilons = np.linspace(0.01, 0.5, n_eps)
    
    mu_matrix = np.zeros((n_mu,n_eps))
    chi_crits = np.zeros((n_mu,n_eps))
    rbs = np.zeros((n_mu,n_eps))
    
    j = 0
    
    for epsilon in epsilons:

        mus = np.linspace(0,4*np.pi*epsilon*Psi/9-1e-6,n_mu)
        mu_matrix[:,j] = mus
        
        i = 0
        
        for mu in mus:
            
            print(f'{Psi}_{epsilon}_{mu}')
            
            chis, crit_chi, crit_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi)
            
            chi_crits[i,j] = crit_chi
            rbs[i,j] = rb
            
            i = i+1            
        j = j+1
        
    results_dict = {'Psi':Psi, 'epsilons':epsilons,'mu_matrix':mu_matrix, 'crit_chis':chi_crits,'rbs':rbs }
    
    fname = f'cmap_Psi_{Psi}_nmus_{n_mu}_neps_{n_eps}'
    
    with open('Data/'+ fname + '.pkl', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return 1

Psis = [3,5,7]
n_mu = 500
n_eps = 500

for Psi in Psis:
    calculate_chi_crits_rbs(Psi,n_mu,n_eps)
    
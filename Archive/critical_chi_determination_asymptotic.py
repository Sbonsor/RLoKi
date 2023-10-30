#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:39:55 2023

@author: s1984454
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
from RLoKi_asymptotics import RLoKi_asymp
from LoKi import LoKi
import pickle 

###Set 1: fix mu/epsilon, vary a0 (Psi, equivalent really)
def fixed_mu_epsilon_panel(mu_epsilon, epsilon, a0_min, a0_max, n_samples_a0, n_samples_chi):

    mu = mu_epsilon * epsilon
    a0s = np.linspace(a0_min, a0_max, n_samples_a0)
    Psis = a0s + (9*mu)/(4*np.pi*epsilon)
    
    chis = np.zeros((n_samples_chi,n_samples_a0))
    flags = np.zeros((n_samples_chi,n_samples_a0))
    
    for i in range(n_samples_a0):
        Psi = Psis[i]
        loki_model = LoKi(mu, epsilon, Psi)
        
        lambda_0 = loki_model.dpsi_dr[-1] * loki_model.rt**2
        alpha_0 = lambda_0/loki_model.rt
        chi_crit_estimate = (-8*alpha_0**3) / (243*lambda_0**2)
        
        chi_range = np.logspace(np.log10(0.1*chi_crit_estimate), np.log10(10*chi_crit_estimate), base = 10.0, endpoint = True, num =n_samples_chi)
        chis[:,i] = chi_range
        
        reached_critical_flag = False
        
        for j in range(n_samples_chi):
            
            if(reached_critical_flag == False):
                chi = chi_range[j]
                
                model = RLoKi_asymp(mu, epsilon, Psi, chi, r_final = 3*loki_model.rt)
                
                psi_equatorial = model.psi_theta_r(np.pi/2)
                
                condition = psi_equatorial <= 0
                
                if(len(psi_equatorial[condition]) == 0):
                    flags[j][i] = 0
                    reached_critical_flag = True
                else:
                    flags[j][i] = 1
                
            else:
                flags[j][i] = 0 
            
            print(flags[j][i])
                    
    return {'a0s' : a0s, 'Psis' : Psis, 'chis' : chis, 'flags' : flags, 'mu' : mu, 'epsilon' : epsilon}

def fixed_Psi_panel(Psi, epsilon, n_samples_mu_eps, n_samples_chi):
    
    mu_epsilon_max = (Psi * 4 * np.pi)/9
    mu_epsilons = np.linspace(0, mu_epsilon_max, n_samples_mu_eps)
    mus = mu_epsilons * epsilon
    
    chis = np.zeros((n_samples_chi,n_samples_mu_eps))
    flags = np.zeros((n_samples_chi,n_samples_mu_eps))
    
    for i in range(n_samples_mu_eps):
        mu = mus[i]
        loki_model = LoKi(mu, epsilon, Psi)
        
        lambda_0 = loki_model.dpsi_dr[-1] * loki_model.rt**2
        alpha_0 = lambda_0/loki_model.rt
        chi_crit_estimate = (-8*alpha_0**3) / (243*lambda_0**2)
        
        chi_range = np.logspace(np.log10(0.1*chi_crit_estimate), np.log10(10*chi_crit_estimate), base = 10.0, endpoint = True, num =n_samples_chi)
        chis[:,i] = chi_range
        
        reached_critical_flag = False
        
        for j in range(n_samples_chi):
            
            if(reached_critical_flag == False):
                chi = chi_range[j]
                
                model = RLoKi_asymp(mu, epsilon, Psi, chi, r_final = 3*loki_model.rt)
                 
                psi_equatorial = model.psi_theta_r(np.pi/2)
                
                condition = psi_equatorial <= 0
                
                if(len(psi_equatorial[condition]) == 0):
                    flags[j][i] = 0
                    reached_critical_flag = True
                else:
                    flags[j][i] = 1
            else:
                flags[j][i] = 0
                    
    return {'mus' : mus, 'Psi' : Psi, 'chis' : chis, 'flags' : flags, 'epsilon' : epsilon}

mu_epsilons = [0,1,2,3]
epsilon = 0.01
a0_min = epsilon**2
a0_max = 7
n_samples_a0 = 50
n_samples_chi = 100
fname1 = 'asymp_panels_1'

set_1_panels = []

for idx in range(len(mu_epsilons)):
    
    results_dict = fixed_mu_epsilon_panel(mu_epsilons[idx], epsilon, a0_min, a0_max, n_samples_a0, n_samples_chi)
    
    set_1_panels.append({'mu_epsilon':mu_epsilons[idx], 'a0_min':a0_min, 'a0_max':a0_max, 'panel': results_dict})

with open(fname1 + '.pkl', 'wb') as f:
    pickle.dump(set_1_panels, f)


Psis = [3,5,7]
epsilon = 0.01
n_samples_mu_eps = 50
n_samples_chi = 100
fname2 = 'asymp_panels_2'

set_2_panels = []

for idx in range(len(Psis)):
    
    results_dict = fixed_Psi_panel(Psis[idx], epsilon, n_samples_mu_eps, n_samples_chi)
    
    set_2_panels.append({'Psi':Psis[idx], 'panel': results_dict})

with open(fname2 + '.pkl', 'wb') as f:
    pickle.dump(set_2_panels, f)





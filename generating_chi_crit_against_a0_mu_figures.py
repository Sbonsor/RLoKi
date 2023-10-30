#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:50:00 2023

@author: s1984454
"""
import numpy as np
import matplotlib.pyplot as plt
from RLoKi import determine_critical_chi
import pickle 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

def crit_chi_against_mu(Psi, epsilon, nsamp, m_iters = 1000, step_size = 0.04, step_size_criterion = 3.90625e-05, calculate = True):
    
    if(calculate == False):
        
        with open('Data/chi_crit_Psi_' + str(Psi) + '_eps_' + str(epsilon) + '.pkl', 'rb') as f:
            results_dict = pickle.load(f)
    
    else:    
        
        mu_max = 4*np.pi*epsilon*Psi/9
        mus = np.linspace(0,mu_max,nsamp)
        
        chi_crits = np.zeros(nsamp)
        
        for i in range(nsamp):
            mu = mus[i]
            chis,last_chi,last_model,rb = determine_critical_chi(mu,epsilon, Psi, m_iters, step_size, step_size_criterion)
            chi_crits[i] = last_chi
            
        results_dict = {'chi_crits' : chi_crits,'mus' : mus, 'Psi_epsilon_miters_stepsize_criterion' : [Psi,epsilon,m_iters,step_size, step_size_criterion]}
        
        with open('Data/chi_crit_Psi_' + str(Psi) + '_eps_' + str(epsilon) + '.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
               
    chi_crits = results_dict['chi_crits']
    mus = results_dict['mus']
    args = results_dict['Psi_epsilon_miters_stepsize_criterion']
    
    fig1,ax1 = plt.subplots()
    ax1.plot(mus, chi_crits)
    ax1.set_xlabel('$\\mu$')
    ax1.set_ylabel('$\\chi_{c}$')
    ax1.set_title('$(\\Psi, \\epsilon) =$ '+ '(' + str(args[0]) + ', '+ str(args[1])+')')
    ax1.set_yscale('log')
      
    return 1

def crit_chi_against_a0(mu, epsilon, a0_min, a0_max, nsamp, m_iters = 1000, step_size = 0.04, step_size_criterion = 3.90625e-05, calculate = True):
    
    if(calculate == False):
        
        with open('Data/chi_crit_mu_' + str(mu) + '_eps_' + str(epsilon) + '.pkl', 'rb') as f:
            results_dict = pickle.load(f)
    
    else:    
        
        a0s = np.linspace(a0_min, a0_max,nsamp)
        Psis = a0s + (9*mu)/(4*np.pi*epsilon)
        
        chi_crits = np.zeros(nsamp)
        
        for i in range(nsamp):
            Psi = Psis[i]
            chis,last_chi,last_model, rb = determine_critical_chi(mu,epsilon, Psi, m_iters, step_size, step_size_criterion)
            chi_crits[i] = last_chi
            
        results_dict = {'chi_crits' : chi_crits,'a0s' : a0s,'Psis': Psis, 'mu_epsilon_miters_stepsize_criterion' : [mu,epsilon,m_iters,step_size, step_size_criterion]}
        
        with open('Data/chi_crit_mu_' + str(mu) + '_eps_' + str(epsilon) + '_n_samp_' + str(nsamp) + '.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
               
    chi_crits = results_dict['chi_crits']
    a0s = results_dict['a0s']
    args = results_dict['mu_epsilon_miters_stepsize_criterion']
    
    fig1,ax1 = plt.subplots()
    ax1.plot(a0s, chi_crits)
    ax1.set_xlabel('$a_0$')
    ax1.set_ylabel('$\\chi_{c}$')
    ax1.set_title('$(\\mu, \\epsilon) =$ '+ '(' + str(args[0]) + ', '+ str(args[1])+')')
    ax1.set_yscale('log')

    
    return 1

Psis = [7]
epsilon = 0.1
m_iters = 10000
step_size = 0.04
step_size_criterion = 3.90625e-05
nsamp = 300
calculate = True

for Psi in Psis:
    crit_chi_against_mu(Psi, epsilon, nsamp, m_iters = m_iters, step_size = step_size, step_size_criterion = step_size_criterion, calculate = calculate)

mus = np.linspace(0,0.3,10)
epsilon = 0.1
a0_min = epsilon**2
a0_max = 1
m_iters = 10000
step_size = 0.04
step_size_criterion = 3.90625e-05
nsamp = 100
calculate = True

for mu in mus:
    print(mu)
    crit_chi_against_a0(mu, epsilon, a0_min, a0_max, nsamp, m_iters = m_iters, step_size = step_size, step_size_criterion = step_size_criterion, calculate = calculate)


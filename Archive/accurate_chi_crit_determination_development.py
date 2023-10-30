#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:08:26 2023

@author: s1984454
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
from RLoKi import RLoKi
from LoKi import LoKi
import pickle 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

def determine_critical_chi(mu,epsilon, Psi, m_iters, step_size, step_size_criterion):
    chis = []
    loki_model = LoKi(mu, epsilon, Psi)
    
    lambda_0 = loki_model.dpsi_dr[-1] * loki_model.rt**2
    alpha_0 = lambda_0/loki_model.rt
    chi_crit_estimate = (-8*alpha_0**3) / (243*lambda_0**2)
    
    chi_n = chi_crit_estimate # 10 ** np.log10(chi_crit_estimate*1.1)
    last_chi = chi_crit_estimate
    
    while(step_size > step_size_criterion):
        chis.append(chi_n)
        model = RLoKi(mu, epsilon, Psi, chi_n, max_iters = m_iters)
        
        if(model.converged == False):
            print('Not converged!')
            step_size = 0.5*step_size
            chi_n = 10 ** (np.log10(chi_n) - step_size)
    
            
        else:
            
            psi_equatorial = model.interp_equatorial_plane()
            
            gradient_estimate = np.gradient(psi_equatorial, model.r_grid)
            
            grad_idx = np.where(gradient_estimate < 0)[0][-1]
            indices_for_interpolation = range(grad_idx-20, min(grad_idx+2, len(gradient_estimate)))
            
            r_grad0 = np.interp(0, gradient_estimate[indices_for_interpolation], model.r_grid[indices_for_interpolation] )
            
            psi_at_turning_point = np.interp(r_grad0, model.r_grid[indices_for_interpolation], psi_equatorial[indices_for_interpolation])
            
            if(psi_at_turning_point > 0):
                step_size = step_size * 0.5
                chi_n = 10 ** (np.log10(chi_n) - step_size)
                
            if(psi_at_turning_point < 0):
                last_chi = chi_n
                chi_n = 10 ** (np.log10(chi_n) + step_size)
                
    return chis, last_chi

# mu = 0.05
# epsilon = 0.01 
# Psi = 7
# m_iters = 10000
# step_size = 0.04
# step_size_criterion = 3.90625e-05

# chis,last_chi = determine_critical_chi(mu,epsilon, Psi, m_iters, step_size, step_size_criterion)

# fig1,ax1 = plt.subplots(1,1)
# ax1.plot(chis)

# critical_model = RLoKi(mu, epsilon, Psi, last_chi, max_iters = m_iters)
# psi_equatorial_crit = critical_model.interp_equatorial_plane()

# fig2,ax2 = plt.subplots(1,1)
# plt.plot(critical_model.r_grid, psi_equatorial_crit)
# ax2.axhline(y=0, color = 'k', linestyle = '--')

########## Looking at single iterations
# chi = 3.816780989102068e-11
# Psi = 5
# epsilon = 0.1
# mu = 0.6911270349034401

chi = 0.00021401252503346906
Psi = 5
epsilon = 0.1
mu = 0.07004665894291623


step_size = 0.04
step_size_criterion = 3.90625e-05 

model = RLoKi(mu, epsilon, Psi, chi)

if(model.converged == False):
    print('Not converged!')
    step_size = 0.5*step_size
    chi_n = chi
    
else:
    
    psi_equatorial = model.interp_equatorial_plane()
    
    gradient_estimate = np.gradient(psi_equatorial, model.r_grid)
    
    grad_idx = np.where(gradient_estimate[1:-1] < 0)[0][-1]
    indices_for_interpolation = range(grad_idx-20, min(grad_idx+2, len(gradient_estimate)))
    
    r_grad0 = np.interp(0, gradient_estimate[indices_for_interpolation], model.r_grid[indices_for_interpolation] )
    
    psi_at_turning_point = np.interp(r_grad0, model.r_grid[indices_for_interpolation], psi_equatorial[indices_for_interpolation])   
    
fig,ax = plt.subplots(1,1)
ax.plot(model.r_grid,psi_equatorial)
ax.plot(model.r_grid, gradient_estimate)
ax.plot(model.r_grid[indices_for_interpolation], gradient_estimate[indices_for_interpolation])
ax.axhline(y=0, color = 'k', linestyle = '--')
ax.axhline(y = psi_at_turning_point, color = 'r', linestyle = '--')
ax.axvline(x = r_grad0, color = 'k', linestyle = '--')
ax.set_ylim(-0.5,0.5)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:00:14 2023

@author: s1984454
"""
import numpy as np
import matplotlib.pyplot as plt
from RLoKi import RLoKi, determine_critical_chi
import pickle


Psi = 5
epsilon = 0.1
mu = 0.3

m_iters = 10000
step_size = 0.04
step_size_criterion = 3.90625e-05

chis, chi_crit, last_model,rb = determine_critical_chi(mu,epsilon, Psi, m_iters, step_size, step_size_criterion)

chis = np.linspace(0,chi_crit, 5, endpoint = True)

fig1,ax1 = plt.subplots(1,1)
max_r = 10
for i in range(len(chis)):
    chi = chis[i]
    model = RLoKi(mu, epsilon, Psi, chi, max_iters = m_iters)
    psi_equatorial = model.interp_equatorial_plane()
    #max_r = max(model.r_grid[-1],max_r) 
    ax1.plot(model.r_grid,psi_equatorial, label = '$\\chi/\\chi_c =$'+ str(i*1/(len(chis)-1)))
    
ax1.set_ylim(-0.1,Psi+1)
ax1.set_xlim(epsilon, rb)
ax1.set_xlabel('$\\hat{r}$')
ax1.set_ylabel('$\\psi$')
ax1.set_title('$(\\Psi,\\epsilon,\\mu)= $' + '(' + str(Psi) + ',' + str(epsilon) + ',' + str(mu) + ')')
ax1.legend()
ax1.axhline(y = 0 , linestyle = '--', color = 'k')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:43:03 2023

@author: s1984454
"""
import numpy as np
import matplotlib.pyplot as plt
from RLoKi import RLoKi,determine_critical_chi
import pickle 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)



Psi = 5
epsilon = 0.1
m_iters = 1000 
step_size = 0.04#3.90625e-05 
step_size_criterion = (3.90625e-05)/16

with open('Data/chi_crit_Psi_' + str(Psi) + '_eps_' + str(epsilon) + '.pkl', 'rb') as f:
    results_dict = pickle.load(f)
    
chi_crits = results_dict['chi_crits']
mus = results_dict['mus']

idx = np.where(chi_crits == min(chi_crits))[0][0]
mu = mus[idx]

chis,last_chi, last_model, rb = determine_critical_chi(mus[idx],epsilon, Psi, m_iters, step_size, step_size_criterion)
psi_equatorial = last_model.interp_equatorial_plane()

fig1,ax1 = plt.subplots(1,1)
ax1.plot(last_model.r_grid, psi_equatorial)
ax1.set_xlabel('$\\hat{r}$')
ax1.set_ylabel('$\\psi$')
ax1.axhline(y = 0, linestyle = '--', color = 'k')
ax1.axvline(x = rb, linestyle = '--', color  = 'r')
# ax1.set_xlim(1100,1400)
# ax1.set_ylim(-2e-5,3e-5)


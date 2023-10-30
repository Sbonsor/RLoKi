#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:33:04 2023

@author: s1984454
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

def generate_cplots_parameter_exploration(Psis, nmus, neps):
    
    fig, ax = plt.subplots(len(Psis),2)

    for i,Psi in enumerate(Psis):
        
        with open(f'Data/cmap_Psi_{Psi}_nmus_{nmus}_neps_{neps}.pkl', 'rb') as f:
            data = pickle.load(f)
        
        crit_chis = data['crit_chis']
        rbs = data['rbs']
    
        mu_matrix = data['mu_matrix']
    
        epsilons = np.reshape(data['epsilons'], (1,neps))
        epsilon_matrix = np.repeat(epsilons, nmus, axis = 0)
    
        
        contour1 = ax[i,0].contourf(epsilon_matrix, mu_matrix, np.log10(crit_chis), levels = 100)
        ax[i,0].set_xlabel('$\\epsilon$')
        ax[i,0].set_ylabel('$\\mu$')
        cbar1 = fig.colorbar(contour1, ax=ax[i,0])
        cbar1.set_label('$\\log_{10}(\\chi_c)$')
        ax[i,0].set_title(f'$\\Psi=$ {Psi}')
    
        contour2 = ax[i,1].contourf(epsilon_matrix, mu_matrix, np.log10(rbs), levels = 100)
        ax[i,1].set_xlabel('$\\epsilon$')
        ax[i,1].set_ylabel('$\\mu$')
        cbar2 = fig.colorbar(contour2, ax=ax[i,1])
        cbar2.set_label('$\\log_{10}(r_B)$')
        ax[i,1].set_title(f'$\\Psi=$ {Psi}')
    
    return 0

Psis = [3,5,7]
nmus = 500
neps = 500

generate_cplots_parameter_exploration(Psis, nmus, neps)

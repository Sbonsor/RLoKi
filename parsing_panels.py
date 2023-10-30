#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:41:44 2023

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


fname = 'panels_1.pkl'

with open('Data/' + fname, 'rb') as f:
    panels = pickle.load(f)

for i in range(len(panels)):    


    current_panel = panels[i]['panel']
    
    chis = current_panel['chis']
    flags = current_panel['flags']
    Psis = current_panel['Psis']
    a0s = current_panel['a0s']
    mu = current_panel['mu']
    epsilon = current_panel['epsilon']
    
    a0s = np.reshape(a0s, (1,len(a0s)))
    a0s = np.repeat(a0s,len(chis), axis = 0)
    
    fig,ax = plt.subplots(1,1)
    ax.set_yscale('log')
    ax.scatter(a0s,chis, marker = 'x', c = flags)
    ax.set_xlabel('$a_0$')
    ax.set_ylabel('$\\chi$')
    ax.set_title('Via iteration: $(\\mu , \\epsilon )=$ ' + '('+str(mu)+','+str(epsilon)+')')
    
fname = 'panels_2.pkl'

with open('Data/' + fname, 'rb') as f:
    panels = pickle.load(f)

for i in range(len(panels)):    

    current_panel = panels[i]['panel']
    
    chis = current_panel['chis']
    flags = current_panel['flags']
    Psi = current_panel['Psi']

    mus = current_panel['mus']
    epsilon = current_panel['epsilon']
    
    mus = np.reshape(mus, (1,len(mus)))
    mus = np.repeat(mus,len(chis), axis = 0)
    
    fig,ax = plt.subplots(1,1)
    ax.set_yscale('log')
    ax.scatter(mus,chis, marker = 'x', c = flags)
    ax.set_xlabel('$\\mu$')
    ax.set_ylabel('$\\chi$')
    ax.set_title('Via iteration: $(\\Psi , \\epsilon )=$ ' + '('+str(Psi)+','+str(epsilon)+')')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:42:05 2023

@author: s1984454
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
from RLoKi import RLoKi
from LoKi import LoKi
import pickle

chi = 2.4414444178099152e-06
Psi = 5
epsilon = 0.1
mu = 0.5


# fname = 'panels_1.pkl'

# with open(fname, 'rb') as f:
#     panels = pickle.load(f)
    
# row = 10
# column = 48
# current_panel = panels[0]['panel']

# chi = current_panel['chis'][row][column]
# Psi = current_panel['Psis'][column]
# epsilon = current_panel['epsilon']
# mu = current_panel['mu']

lmax = 10
leg_threshold = 1e-10
m_iters = 100

# loki_model = LoKi(mu, epsilon, Psi)
# lambda_0 = loki_model.dpsi_dr[-1] * loki_model.rt**2
# alpha_0 = lambda_0/loki_model.rt
# chi_crit_estimate = (-8*alpha_0**3) / (243*lambda_0**2)
# chi_range = np.logspace(np.log10(0.1*chi_crit_estimate), np.log10(10*chi_crit_estimate), base = 10.0, endpoint = True, num = 3)


model = RLoKi(mu, epsilon, Psi, chi, legendre_threshold = leg_threshold, max_iters = m_iters, lmax = lmax)

psi = model.psis
rho_hat_coeffs = model.rho_hat_coeffs

# for i in range(len(psi)):
#     plt.plot(model.r_grid, psi[i][5,:])
    
fig, ax = plt.subplots(1,1)
ax.set_yscale('log')
for l in range(lmax):
    coeffs = rho_hat_coeffs[0][l,:]
    ax.plot(model.r_grid, abs(coeffs), label = f'$l$ = {l}')
ax.legend()
ax.set_xlabel('$r$')
ax.set_ylabel('$\\rho_l^{(0)}(r)$')

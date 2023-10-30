#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:47:00 2023

@author: s1984454
"""
import numpy as np
from numpy.polynomial.legendre import legval
import flt
from scipy.special import gammainc,gamma
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator
from LoKi import LoKi
from RLoKi import RLoKi
import matplotlib.pyplot as plt


Psi = 5
mu = 0
epsilon = 0.01
chi = 0

model = RLoKi(mu, epsilon, Psi, chi)

base_model = LoKi(mu, epsilon, Psi, pot_only = True)

fig1,ax1 = plt.subplots(1,1)
ax1.axhline(y=0, linewidth = 0.5, color = 'k', linestyle = '--')
ax1.set_xlabel('$\\hat{r}$')
ax1.set_ylabel('$\\psi(\\hat{r})$')
ax1.set_title('$(\\Psi,\\chi,\\epsilon,\\mu)=$'+'('+str(Psi)+','+str(chi)+','+str(epsilon)+','+str(mu)+')')
ax1.plot(base_model.rhat, base_model.psi, label = 'LoKi model')
ax1.plot(model.r_grid, model.psi[5,:], label = 'Iteration scheme')
ax1.legend()
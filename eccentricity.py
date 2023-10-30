#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:53:11 2023

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

def zero_crossing(f,r):

    first_negative = np.where(f<0)[0][0]
    indices_for_interpolation = range(max(0,first_negative-10), min(len(f), first_negative + 10))
    zero_crossing = np.interp(0,f[indices_for_interpolation][::-1],r[indices_for_interpolation][::-1])
    
    return zero_crossing

def determine_eccentricity_profile(equatorial_profile, polar_profile, r_grid, npoints):
    
    ahats = np.linspace(0.1, zero_crossing(equatorial_profile,r_grid) - 0.1, npoints)
    bhats = np.zeros(len(ahats))
    e  = np.zeros(len(bhats))

    for i in range(len(bhats)):
        ahat = ahats[i]
        psi = np.interp(ahat, r_grid, equatorial_profile)
        bhats[i] = np.interp(psi, polar_profile[::-1], r_grid[::-1])
        
    e = np.sqrt(1-bhats**2/ahats**2)
    
    
    return e, bhats, ahats

Psi = 5
mu = 0.2
epsilon = 0.1

loki_model = LoKi(mu, epsilon, Psi)


###### Via asymptotics
chis, chi_crit, asymp_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi, 3.90625e-05)
print(chi_crit)
polar_psi = asymp_model.psi_theta_r(0)
equatorial_psi = asymp_model.psi_theta_r(np.pi/2)
rhat = asymp_model.rhat

RLoKi_model =RLoKi(mu, epsilon, Psi, chi_crit)

# fig2,ax2 = plt.subplots(1,1)
# #ax2.set_title('By Asymptotics')
# ax2.plot(rhat,equatorial_psi, label = 'Asymptotics Equatorial', color = 'k')
# ax2.plot(rhat,polar_psi, label = 'Asymptotics Polar', color = 'k', linestyle = '--')
# ax2.plot(RLoKi_model.r_grid, RLoKi_model.interp_to_theta(0), label = 'Iteration Polar', color = 'r', linestyle = '--', alpha = 0.5)
# ax2.plot(RLoKi_model.r_grid, RLoKi_model.interp_to_theta(np.pi/2), label = 'Iteration Equatorial', color = 'r', alpha = 0.5)
# ax2.set_title(f'$(\\Psi,\\mu,\\epsilon)$ = ({Psi},{mu},{epsilon})')
# ax2.legend(loc = 'upper right')
# #ax2.axhline(y= polar_psi[0])
# #ax2.axhline(y = 0, linestyle = '--', color = 'k')
# #ax2.plot(loki_model.rhat, loki_model.psi, label = 'Non-rotating')


# e1, bhats1, ahats1 = determine_eccentricity_profile(equatorial_psi, polar_psi, rhat, 100)
    
# central_e_estimate = np.sqrt( 1 - (-1.5 + chi_crit *(3 - 0.5 * asymp_model.A_2)) / (-1.5 + chi_crit *(3 + asymp_model.A_2)))

# fig3,ax3 = plt.subplots(1,1)
# ax3.set_title(f'$(\\Psi,\\mu,\\epsilon)$ = ({Psi},{mu},{epsilon})')
# ax3.plot(bhats1[1:], e1[1:], label = 'Asymptotics')
# ax3.set_xlabel('$\\hat{b}$')
# ax3.set_ylabel('eccentricity')
#ax3.axhline(y = central_e_estimate, linestyle = '--', color = 'k')


# ##### Via iteration
# chis, last_chi, model, rb = determine_critical_chi(mu,epsilon, Psi, 1000, 0.04, 3.90625e-05)
# print(last_chi)
# #model = RLoKi(mu, epsilon, Psi, chi_crit)
# polar_psi = model.interp_to_theta(0)
# equatorial_psi = model.interp_to_theta(np.pi/2)
# rhat = model.r_grid

# # fig4,ax4 = plt.subplots(1,1)
# # ax4.set_title('By Iteration')
# ax2.plot(rhat,equatorial_psi, label = 'Iteration Equatorial')
# ax2.plot(rhat,polar_psi, label = 'Iteration Polar')
# # ax4.legend()
# # ax4.axhline(y = 0, linestyle = '--', color = 'k')

# e2, bhats2, ahats2 = determine_eccentricity_profile(equatorial_psi, polar_psi, rhat, 100)
    
# ax3.plot(bhats2, e2, label = 'Iteration')
# ax3.legend()
# ax2.legend()


epsilon = 0.1
Psi = 5
mu_crit = 0.6907869873507858
mus_frac = [0,0.3,1]
linestyles = ['solid', 'dotted', 'dashdot']
fig4, axis = plt.subplots(1,1)

for i in range(len(mus_frac)):
    
    mu = mus_frac[i]*mu_crit
    #axis = axes[i] 
    
    chis, crit_chi, crit_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi)
    polar_psi = crit_model.psi_theta_r(0)
    equatorial_psi = crit_model.psi_theta_r(np.pi/2)
    rhat = crit_model.rhat
    e1, bhats1, ahats1 = determine_eccentricity_profile(equatorial_psi, polar_psi, rhat, 100)

    axis.set_title(f'$(\\Psi,\\epsilon)$ = ({Psi},{epsilon})')
    axis.plot(ahats1[1:]/rb, e1[1:], label = f'$\\mu/\\mu_c$ = {mus_frac[i]}', linestyle = linestyles[i])
    axis.set_xlabel('$\\hat{a}/r_B$')
    axis.set_ylabel('e')
    axis.legend(loc = 'upper left')







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 09:17:18 2023

@author: s1984454
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
from RLoKi import RLoKi,determine_critical_chi
from RLoKi_asymptotics import RLoKi_asymp,determine_critical_chi_asymp
from LoKi import LoKi
from LoKi_asymptotics import LoKi_asymp
import pickle 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

##### Eccentricity plot, fig.
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
    
##### Density contour plots, fig.

def generate_contour_plot_asymp(model, axlim, n_angle, epsilon, mu_frac, Psi, axis,rb):
    
    mu = model.mu
    epsilon = model.epsilon
    Psi = model.Psi
    
    theta_grid = np.linspace(0,np.pi,n_angle)
    rhat = model.rhat
    axlim = axlim/rb
    psi = np.zeros((len(theta_grid),len(rhat)))
    
    i = 0
    for theta in theta_grid:
        psi[i,:] = model.psi_theta_r(theta)
        i = i+1
    R,THETA = np.meshgrid(rhat/rb,theta_grid)
    
    X = R*np.sin(THETA)
    Y = R*np.cos(THETA)
    
    axis.contour(X, Y, psi, levels = 20, colors = 'k', linewidths = 1, vmin = 0)
    axis.set_xlabel('$\\hat{r}/r_B$')
    axis.set_ylabel('$\\hat{z}/r_B$')
    axis.set_xlim(0,axlim)
    axis.set_ylim(-axlim,axlim)
    axis.set_aspect('equal')
    axis.set_title(f'({Psi}, {mu_frac}, {epsilon})')

        
    return 1

epsilon = 0.1
Psis = [3,5]
mu_crits = [0.4030162313714636,0.6907869873507858]
mus_frac = [0,0.3,1]

# epsilon = 0.1
# Psis = [5]
# mu_crits = [0.6907869873507858]
# mus_frac = [0.3]


nrows, ncols = len(Psis), len(mus_frac)
dx, dy = 1, 2
figsize = plt.figaspect(float(dy * nrows) / float(dx * ncols))
fig4, axes = plt.subplots(nrows,ncols, figsize = figsize)

for j in range(len(Psis)):
    
    Psi = Psis[j]
    mu_crit = mu_crits[j]
    
    for i in range(len(mus_frac)):
        
        mu = mus_frac[i]*mu_crit
        
        if(nrows==1 and ncols==1):
            axis = axes
        else:
            axis = axes[j,i] 
        
        chis, crit_chi, crit_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi)
        generate_contour_plot_asymp(crit_model, rb, 100, epsilon, mus_frac[i], Psi, axis,rb)
        
##### Chi_crit example, fig.

def psi_profiles_fixed_Psi_epsilon_mu(Psi, mu, epsilon, num_chis):
    _, chi_crit, _, r_b = determine_critical_chi_asymp(mu,epsilon, Psi, step_size_criterion =  3.90625e-05/16)
    
    chis = np.linspace(0,chi_crit,num_chis)
    
    fig,ax = plt.subplots(1,1)
    ax.set_ylabel('$\\psi$')
    ax.set_xlabel('$\\hat{r}$')
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    linestyles = ['solid', 'dotted', 'dashdot', 'dashed', (0, (5, 1))]
    
    i=0
    for chi in chis:
        
        model = RLoKi_asymp(mu, epsilon, Psi, chi)
        ax.plot(model.rhat, model.psi_theta_r(np.pi/2), label = str(i/(num_chis-1)) +'$\\chi_{c}$', linestyle = linestyles[i])
        axins.plot(model.rhat, model.psi_theta_r(np.pi/2), linestyle = linestyles[i])
        i += 1
        
    ax.set_xlim(epsilon,model.rhat[-1])    
    ax.set_ylim(1e-8,Psi)
    ax.set_title(f'$(\\Psi, \\mu, \\epsilon) = ({Psi},{mu},{epsilon})$')
    ax.legend(loc = 'lower left')
    
    #ax.set_xscale('log')
    #ax.set_yscale('log')
     
    # subregion of the original image
    x1, x2, y1, y2 = 9, 15, 0, 0.6
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    return 1

Psi = 5
mu = 0
epsilon = 0.1
num_chis = 5

psi_profiles_fixed_Psi_epsilon_mu(Psi, mu, epsilon, num_chis)

##### chi_crit vs a_0, fig.

#mus = [0, 0.1, 0.3]
mus = np.concatenate((np.array([0,1e-6]) , np.linspace(0.03, 0.3, 9)))
epsilon = 0.1
#linestyles = ['solid', 'dotted', 'dashdot']

fig,ax = plt.subplots(1,1)
ax.set_xlabel('$a_0$')
ax.set_ylabel('$\\chi_{c}$')
ax.set_yscale('log')

i=0
for mu in mus:
    
    with open('Data/asymp_chi_crit_mu_' + str(mu) + '_eps_' + str(epsilon) + '_a0max_1.pkl', 'rb') as f:
        results_dict = pickle.load(f)

    chi_crits = results_dict['chi_crits']
    a0s = results_dict['a0s']
    print(a0s[1]-a0s[0])
    print(min(a0s))
    args = results_dict['mu_epsilon_stepsize_criterion']
    
    ax.plot(a0s, chi_crits, label = f'$\\mu$ = {mu}')#, linestyle = linestyles[i])
    ax.legend(loc = 'upper right')
    i=i+1

##### chi_crit vs mu, fig.

Psis = [3,5,7]
epsilon = 0.01
linestyles = ['solid', 'dotted', 'dashdot']

fig,ax = plt.subplots(1,1)
ax.set_xlabel('$\\mu/\\mu_{max}$')
ax.set_ylabel('$\\chi_{c}$')
ax.set_yscale('log')
ax.set_xlim(0,1)

i = 0
for Psi in Psis:
    
    with open('Data/asymp_chi_crit_Psi_' + str(Psi) + '_eps_' + str(epsilon) + '.pkl', 'rb') as f:
        results_dict = pickle.load(f)
        
    chi_crits = results_dict['chi_crits']
    mus = results_dict['mus']
    mu_max = 4*np.pi*epsilon*Psi/9
    ax.plot(mus/mu_max,chi_crits, label = f'$\\Psi$ = {Psi}', linestyle = linestyles[i])
    ax.legend(loc = 'lower left')
    i = i+1

#### Single critical LoKi model
Psi = 5
mu = 0.6907869873507858
epsilon = 0.1

model = LoKi(mu, epsilon, Psi)
asymp = LoKi_asymp(model)

fig,ax = plt.subplots(1,1)
ax.plot(model.rhat, model.psi, label = 'Numerical solution')
ax.plot(asymp.r_1_regime_III * epsilon, asymp.psi_regime_III_region_1, label = 'Region I solution', linestyle = 'dashed')
ax.plot(asymp.r_2_regime_III * epsilon**(-3), asymp.psi_regime_III_region_2, label = 'Region II solution', linestyle = 'dotted')
ax.legend(loc = 'lower left')
ax.set_xlabel('r')
ax.set_ylabel('$\\psi$')
ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_xlim(epsilon, model.rt)
ax.set_ylim(1e-6,Psi)
ax.set_title(f'$(\\Psi, \\mu, \\epsilon)$ = ({Psi}, $\\mu_c$, {epsilon})')
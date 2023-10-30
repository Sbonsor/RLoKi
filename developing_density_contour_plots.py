#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:50:03 2023

@author: s1984454
"""
import numpy as np
import matplotlib.pyplot as plt
from RLoKi import RLoKi, determine_critical_chi
from RLoKi_asymptotics import determine_critical_chi_asymp, RLoKi_asymp
from scipy.interpolate import RegularGridInterpolator
import pickle
from scipy.special import gammainc, gamma

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

def generate_contour_plot(psi, r_grid, theta_grid, fname, axlim, n_angle = None):
    
    if(n_angle == None):
        new_theta_grid = theta_grid
    else:
        new_theta_grid = np.linspace(0,np.pi,n_angle)

    psi_interp = np.zeros((len(new_theta_grid),len(r_grid)))
    
    for i in range(len(new_theta_grid)):
        interpolator = RegularGridInterpolator((theta_grid, r_grid), psi)
    
        psi_interp[i,:] =  interpolator((new_theta_grid[i], r_grid))
    
    
    R,THETA = np.meshgrid(r_grid,new_theta_grid)
    
    X = R*np.sin(THETA)
    Y = R*np.cos(THETA)
    
    fig,ax = plt.subplots()
    
    ax.contour(X, Y, psi_interp, levels = 12, colors = 'k', linewidths = 1)
    ax.set_xlabel('$\\hat{r}$')
    ax.set_ylabel('$\\hat{z}$')
    ax.set_xlim(0,axlim)
    ax.set_ylim(-axlim,axlim)
    ax.set_aspect('equal')
    
    with open('figures/' + fname + '.png' , 'wb') as f:
        plt.savefig(f)
        
    return fig,ax

def generate_contour_plot_asymp(model, axlim, n_angle, epsilon, mu_frac, Psi, axis):
    
    mu = model.mu
    epsilon = model.epsilon
    Psi = model.Psi
    
    theta_grid = np.linspace(0,np.pi,n_angle)
    rhat = model.rhat
    psi = np.zeros((len(theta_grid),len(rhat)))
    
    i = 0
    for theta in theta_grid:
        psi[i,:] = model.psi_theta_r(theta)
        i = i+1
    R,THETA = np.meshgrid(rhat,theta_grid)
    
    X = R*np.sin(THETA)
    Y = R*np.cos(THETA)
    
    axis.contour(X, Y, psi, levels = 20, colors = 'k', linewidths = 1)
    axis.set_xlabel('$\\hat{r}$')
    axis.set_ylabel('$\\hat{z}$')
    axis.set_xlim(0,axlim)
    axis.set_ylim(-axlim,axlim)
    axis.set_aspect('equal')
    axis.set_title(f'$(\\Psi,\\mu/\\mu_c,\\epsilon)=$({Psi},{mu_frac},{epsilon})')

        
    return 1

# mu = 0.2
# epsilon = 0.1
# Psi = 5
# fname = f'Psi_{Psi}_mu_{mu}_epsilon{epsilon}_density_contours.png'
# # m_iters = 1000
# # step_size = 0.04
# # step_size_criterion = 3.90625e-05

# # chis, critical_chi, critical_model, rb = determine_critical_chi(mu,epsilon, Psi, m_iters, step_size, step_size_criterion)

# # psi = critical_model.psi
# # theta_grid = critical_model.theta_grid
# # r_grid = critical_model.r_grid

        
# # fig1,ax1 = generate_contour_plot(psi, r_grid, theta_grid, 'refined_contour',rb, 200)


# chis, crit_chi, crit_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi)
# fig2,ax2 = generate_contour_plot_asymp(crit_model, fname,rb, 100,epsilon, mu, Psi)

epsilon = 0.1
Psi = 3
alpha_c = 25.090180360721448
D_Psi = 60.13122960833789
C_Psi = -8.893682287340312
kappa = 18/(5*np.exp(Psi)*gamma(5/2)*gammainc(5/2,Psi)) 
A_2 = alpha_c-D_Psi- 40*np.power(kappa,2)*np.power(Psi,4)*np.log(epsilon)
a_0 = (A_2*np.power(epsilon,2)-C_Psi)*np.power(epsilon,2)
mu = (Psi-a_0)*(4*np.pi*epsilon)/9
# fname = f'Psi_{Psi}_mu_{mu}_epsilon{epsilon}_density_contours.png'
# chis, crit_chi, crit_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi)
# fig3,ax3 = generate_contour_plot_asymp(crit_model, fname,rb, 100,epsilon, mu, Psi)



epsilon = 0.1
Psi = 5
mu_crit = 0.6907869873507858
mus_frac = [0,0.3,1]

fig4, axes = plt.subplots(1,3)

for i in range(len(mus_frac)):
    
    mu = mus_frac[i]*mu_crit
    axis = axes[i] 
    
    chis, crit_chi, crit_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi)
    generate_contour_plot_asymp(crit_model, rb, 100, epsilon, mus_frac[i], Psi, axis)











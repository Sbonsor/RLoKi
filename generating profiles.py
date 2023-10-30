#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:09:29 2023

@author: s1984454
"""

import numpy as np
import matplotlib.pyplot as plt
from RLoKi_asymptotics import determine_critical_chi_asymp, RLoKi_asymp
from LoKi_asymptotics import determine_critical_alpha
from scipy.special import gammainc, gamma
from scipy.stats import linregress
from scipy.integrate import solve_ivp

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

### fixed Psi,epsilon,mu, increasing chi

def psi_profiles_fixed_Psi_epsilon_mu(Psi, mu, epsilon, num_chis):
    _, chi_crit, _, r_b = determine_critical_chi_asymp(mu,epsilon, Psi, step_size_criterion =  3.90625e-05/16)
    
    chis = np.linspace(0,chi_crit,num_chis)
    
    fig,ax = plt.subplots(1,1)
    ax.set_ylabel('$\\psi$')
    ax.set_xlabel('$\\hat{r}$')
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    
    i=0
    for chi in chis:
        
        model = RLoKi_asymp(mu, epsilon, Psi, chi)
        ax.plot(model.rhat, model.psi_theta_r(np.pi/2), label = str(i/(num_chis-1)) +'$\\chi_{c}$')
        axins.plot(model.rhat, model.psi_theta_r(np.pi/2))
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

### Regime I values

Psi = 5
mu = 0
epsilon = 0.1
num_chis = 5

psi_profiles_fixed_Psi_epsilon_mu(Psi, mu, epsilon, num_chis)

### Regime III values

# Psi = 5
# epsilon = 0.1

# alpha_c, C_Psi, D_Psi,kappa = determine_critical_alpha(Psi, alpha_min = -40 , alpha_max = 100, plot = False)

# mu_c = (4*np.pi*epsilon/9)*(Psi + np.power(epsilon,2)*C_Psi + 40*np.power(kappa,2)*np.power(Psi,4)*np.power(epsilon,4)*np.log(epsilon) - np.power(epsilon,4)*(alpha_c-D_Psi))     

# psi_profiles_fixed_Psi_epsilon_mu(Psi, mu_c, epsilon, num_chis)

# ### fixed a_0, epsilon, chi, increasing mu

def rho_hat(psi):
    
    density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi) 
 
    return density
   
def psi_profiles_fixed_a0_epsilon_critical_chi(a_0,Psi_max,epsilon,n_Psis):
    
    Psis = np.linspace(a_0,Psi_max,n_Psis)
    
    chi_crits = []
    scaled_chi_crits = []
    r_bs = []
    
    fig,ax = plt.subplots(1,1)
    ax.set_ylabel('$\\psi$')
    ax.set_xlabel('$\\hat{r}$')
    
    for Psi in Psis:
        
        mu = (Psi-a_0)*(4*np.pi*epsilon/9)
        
        _, chi_crit, _, r_b = determine_critical_chi_asymp(mu,epsilon, Psi, step_size_criterion =  3.90625e-05/16)
        chi_crits.append(chi_crit)
        scaled_chi_crits.append( chi_crit * rho_hat(Psi) )
        r_bs.append(r_b)
        
        model = RLoKi_asymp(mu, epsilon, Psi, chi_crit)
        
        rtild = model.rhat * 1/np.sqrt(rho_hat(Psi))
        ax.plot(rtild, model.psi_theta_r(np.pi/2), label = f'$\\Psi = ${Psi}' )
        
    ax.set_xlim(epsilon,model.rhat[-1])    
    ax.set_ylim(1e-8,Psi)
    ax.set_title(f'$(\\Psi, \\mu, \\epsilon) = ({Psi},{mu},{epsilon})$')
    ax.legend(loc = 'lower left')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    def regime_I_region_II_scaled(r,y):
        
        RHS= np.zeros(2,)
        RHS[0] = y[1]
        RHS[1] = -(2/r)*y[1] - 9*rho_hat(y[0])

        return RHS

    def zero_cross(r,y):
        return y[0]  
    zero_cross.terminal = True

    def solve_regime_I_region_II_scaled(a_0, final_radius = 1e9 ):
        
        solution = solve_ivp(fun = regime_I_region_II_scaled, t_span = (1e-6,1e8), y0 = (a_0,0),method = 'RK45', dense_output = True,rtol = 1e-8, atol = 1e-30,events = zero_cross)
        
        psi_tild = solution.y[0,:]
        psi_tild_grad = solution.y[1,:]
        r_tild = solution.t
        
        return psi_tild, r_tild, psi_tild_grad
    
    psi_tild, r_tild, psi_tild_grad = solve_regime_I_region_II_scaled(a_0, final_radius = 1e9 )
    
    alpha_0_tild = psi_tild_grad[-1] * r_tild[-1]**2
    lambda_0_tild = alpha_0_tild/ r_tild[-1]
    chi_c_tild = -8 * alpha_0_tild**3/(243*lambda_0_tild**2)
    ax.plot(r_tild, psi_tild + (9/2)* chi_c_tild * r_tild**2, color = 'r')
    
    return scaled_chi_crits, Psis, chi_crits, r_bs




a_0 = 3
Psi_max = 9 
epsilon = 0.1
n_Psis = 5

scaled_chi_crits, Psis, chi_crits,r_bs = psi_profiles_fixed_a0_epsilon_critical_chi(a_0,Psi_max,epsilon,n_Psis)

def fit_log(x,y):
    
    x = np.log10(x)
    y = np.log10(y)
    result = linregress(x,y)
    slope = result[0]
    intercept = result[1]
    
    fitted_line = 10**(slope*x + intercept)

    return slope, intercept, fitted_line

# fig,ax = plt.subplots(1,1)
# ax.plot(Psis, scaled_chi_crits, label = 'Scaled')
# ax.plot(Psis,chi_crits, label = 'Unscaled')
# ax.set_ylabel('$\\chi_{crit}$')
# ax.set_xlabel('$\\Psi$')
# ax.set_yscale('log')

slope, intercept, fitted_line = fit_log(rho_hat(Psis),chi_crits)


    
    
    
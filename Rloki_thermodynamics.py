#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:24:27 2023

@author: s1984454
"""
import numpy as np
import matplotlib.pyplot as plt
from RLoKi_asymptotics import RLoKi_asymp,determine_critical_chi_asymp
from RLoKi import RLoKi
from scipy.integrate import simpson, romb
from LoKi import LoKi
from scipy.special import gammainc,gamma
from scipy.stats import linregress
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)


def asymp_M_hat(model):
    
    M_hat = (-4*np.pi*model.rho_hat(model.Psi))/9 * ( (9*mu)/(4*np.pi) + model.lambda_0 + model.chi * model.lambda_1)
    
    return M_hat

def M_hat_naive(model, n_theta):

    Psi = model.Psi
    epsilon = model.epsilon
    chi = model.chi
    mu = model.mu
    
    n_rhat = len(model.rhat)
    theta_grid = np.linspace(0,np.pi,n_theta)
    integrand_2d = np.zeros((n_theta, n_rhat))
    
    for i, theta in enumerate(theta_grid):
    
        psi = model.psi_theta_r(theta)
        rhat = model.rhat
        rho_hat = np.array(list(map(model.rho_hat,psi)))
        
        integrand_2d[i,:] = rho_hat * rhat**2 * np.sin(theta)
        
    M_hat = 2 * np.pi * simpson(y = simpson(y = integrand_2d, x = rhat, axis = -1), x = theta_grid, axis = 0)
    
    return M_hat

def M_hat_iteration(model, n_theta):

    Psi = model.Psi
    epsilon = model.epsilon
    chi = model.chi
    mu = model.mu
    
    n_rhat = len(model.r_grid)
    theta_grid = np.linspace(0,np.pi,n_theta)
    integrand_2d = np.zeros((n_theta, n_rhat))
    
    for i, theta in enumerate(theta_grid):
    
        psi = model.interp_to_theta(theta)
        rhat = model.r_grid
        rho_hat = np.array(list(map(model.rho_hat,psi)))
        
        integrand_2d[i,:] = rho_hat * rhat**2 * np.sin(theta)
        
    M_hat = 2 * np.pi * simpson(y = simpson(y = integrand_2d, x = rhat, axis = -1), x = theta_grid, axis = 0)
    
    return M_hat


def M_hat_naive2(model, n_theta, n_rhat):

    Psi = model.Psi
    epsilon = model.epsilon
    chi = model.chi
    mu = model.mu
    
    rhat = np.logspace(np.log10(epsilon), np.log10(max(model.rhat)), n_rhat)
    theta_grid = np.linspace(0,np.pi,n_theta)
    
    integrand_2d = np.zeros((n_theta, n_rhat))
    
    for i, theta in enumerate(theta_grid):
    
        psi = np.interp(x = rhat, xp = model.rhat, fp = model.psi_theta_r(theta), left = 0, right = 0)
        
        rho_hat = np.array(list(map(model.rho_hat,psi)))
        
        integrand_2d[i,:] = rho_hat * rhat**2 * np.sin(theta)
        
    M_hat = 2 * np.pi * simpson(y = simpson(y = integrand_2d, x = rhat, axis = -1), x = theta_grid, axis = 0)
    
    return M_hat

def W_hat_naive(model, n_theta):
    
    Psi = model.Psi
    epsilon = model.epsilon
    chi = model.chi
    mu = model.mu

    n_rhat = len(model.rhat)
    theta_grid = np.linspace(0,np.pi,n_theta)
    integrand_2d = np.zeros((n_theta, n_rhat))

    for i, theta in enumerate(theta_grid):

        psi = model.psi_theta_r(theta)
        rhat = model.rhat
        rho_hat = np.array(list(map(model.rho_hat,psi)))
        
        integrand_2d[i,:] = rho_hat * (psi - (9/2) * chi * rhat**2 * np.sin(theta)**2 - (9 * mu)/(4 * np.pi * rhat)  ) * rhat**2 * np.sin(theta)
        
        integral = np.pi * simpson(y = simpson(y = integrand_2d, x = rhat, axis = -1), x = theta_grid, axis = 0)
        
    W_hat = (model.alpha_0 + chi*model.alpha_1) * 0.5 * asymp_M_hat(model) - integral
    
    return W_hat

def W_hat_naive2(model, n_theta, n_rhat):
    
    Psi = model.Psi
    epsilon = model.epsilon
    chi = model.chi
    mu = model.mu

    rhat = np.logspace(np.log10(epsilon), np.log10(max(model.rhat)), n_rhat)
    theta_grid = np.linspace(0,np.pi,n_theta)
    
    integrand_2d = np.zeros((n_theta, n_rhat))

    for i, theta in enumerate(theta_grid):

        psi = np.interp(x = rhat, xp = model.rhat, fp = model.psi_theta_r(theta), left = 0, right = 0)
        
        rho_hat = np.array(list(map(model.rho_hat,psi)))
        
        integrand_2d[i,:] = rho_hat * (psi - (9/2) * chi * rhat**2 * np.sin(theta)**2 - (9 * mu)/(4 * np.pi * rhat)  ) * rhat**2 * np.sin(theta)
        
        integral = np.pi * simpson(y = simpson(y = integrand_2d, x = rhat, axis = -1), x = theta_grid, axis = 0)
        
    W_hat = (model.alpha_0 + chi*model.alpha_1) * 0.5 * asymp_M_hat(model) - integral
    
    return W_hat


def combined_beta_E(model, n_theta, n_rhat):
    
    Psi = model.Psi
    epsilon = model.epsilon
    chi = model.chi
    mu = model.mu

    rhat = np.logspace(np.log10(epsilon), np.log10(max(model.rhat)), n_rhat)
    theta_grid = np.linspace(0,np.pi,n_theta)
    
    integrand2_2d = np.zeros((n_theta, n_rhat))
    
    for i, theta in enumerate(theta_grid):

        psi = np.interp(x = rhat, xp = model.rhat, fp = model.psi_theta_r(theta), left = 0, right = 0)
        
        rho_hat = np.array(list(map(model.rho_hat,psi)))

        M_hat = asymp_M_hat(model)
        
        integrand2_2d[i,:] = rho_hat * (psi - (9/2) * chi * rhat**2 * np.sin(theta)**2 - (9 * mu)/(4 * np.pi * rhat)  ) * rhat**2 * np.sin(theta)    
        integral2 = np.pi * simpson(y = simpson(y = integrand2_2d, x = rhat, axis = -1), x = theta_grid, axis = 0)
        
    W_hat = (model.alpha_0 + chi*model.alpha_1) * 0.5 * M_hat  - integral2
    
    beta = ((81 * 3**(2/3))/(16 * np.pi**2 * model.rho_hat(Psi)**2)) * M_hat**(4/3)
    
    E = ((-8 * np.pi**2 * model.rho_hat(Psi)**2) / (9*3**(8/3))) * (W_hat/(M_hat**(7/3))) 
    
    
    return beta, E

def old_beta_energy(mu,epsilon, Psi):
    
    non_loaded_model = LoKi(mu, epsilon, Psi)
    old_beta = ((81 * 3**(2/3))/(16*np.pi**2) ) * (non_loaded_model.M_hat**(4/3)) / non_loaded_model.density(Psi)**(2/3)
    old_energy = -2*np.pi*non_loaded_model.density(Psi)**(2/3)*non_loaded_model.U_hat / (3**(8/3) * non_loaded_model.M_hat**(7/3))
    
    return old_beta, old_energy

######## Testing asymptotic mass approximation against a brute force integration
# epsilon = 0.1
# mu = 0.1
# Psi = 5
# n_chis = 50
# n_theta = 100
# n_rhat = 1000

# chis, crit_chi, crit_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi)

# chis = np.linspace(0, crit_chi, n_chis)

# M_by_asymp = np.zeros(n_chis)
# M_naive = np.zeros(n_chis)
# M_iteration = np.zeros(n_chis)

# for i, chi in enumerate(chis):
    
#     model = RLoKi_asymp(mu, epsilon, Psi, chi)
#     model_iter = RLoKi(mu, epsilon, Psi, chi)
    
#     M_by_asymp[i] = asymp_M_hat(model)
#     M_naive[i] = M_hat_naive2(model, n_theta, n_rhat)
#     M_iteration[i] = M_hat_iteration(model_iter, n_theta)
#     print(i)
    
# relative_errors = abs(M_by_asymp - M_naive)/M_naive
# iter_rel_errors = abs(M_by_asymp - M_iteration)/M_iteration

# fig, ax  = plt.subplots(1,1)
# ax.plot(chis, relative_errors, marker = 'x')
# ax.plot(chis, iter_rel_errors, marker = 'x')
# ax.set_xlabel('$\\chi$')
# ax.set_ylabel('Relative error in mass calculation')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_title(f'$(\\Psi, \\epsilon, \\mu)$ = ({Psi}, {epsilon}, {mu})')


########### First generate the usual classical king spiral
# chi = 0
# epsilon = 0.1
# mu = 0.3

# n_theta = 100
# n_rhat = 1000

# a0s = np.concatenate((np.linspace(0.05, 0.2, 300), np.linspace(0.2, 25, 450)))
# #a0s = np.linspace(0.1,15,100)

# betas_base = np.zeros(len(a0s))
# Es_base = np.zeros(len(a0s))

# old_betas = np.zeros(len(a0s))
# old_energies = np.zeros(len(a0s))

# for i, a0 in enumerate(a0s):
    
#     Psi = a0 + (9*mu)/(4*np.pi * epsilon)
    
#     model = RLoKi_asymp(mu, epsilon, Psi, chi)
#     betas_base[i], Es_base[i] =  combined_beta_E(model, n_theta, n_rhat)
#     old_betas[i], old_energies[i] = old_beta_energy(mu,epsilon, Psi)
#     print(i)

# fig, ax = plt.subplots(1,1)
# ax.plot(Es_base, betas_base, label = '$\\chi$ = 0')#, marker = 'x')
# #ax.plot(old_energies, old_betas, label = 'Varying $\\Psi$', marker = 'x')
# ax.set_xlabel('$\\mathcal{E}$')
# ax.set_ylabel('$\\beta$')
# #ax.set_title(f'$(\\mu, \\chi, \\epsilon)$ = ({mu}, {chi}, {epsilon})')
# ax.set_xlim(0,2)
# ax.set_ylim(0,1.7)

########### Put a track on the diagram for a sequence of models with increasing chi for a 

# Psi = 4
# epsilon = 0.1    
# mu = 0.1

# _, crit_chi, _,_ = determine_critical_chi_asymp(mu,epsilon, Psi)

# chis = np.linspace(0,crit_chi,100)

# betas1 = np.zeros(len(a0s))
# Es1 = np.zeros(len(a0s))

# for i, chi in enumerate(chis):
    
#     model = RLoKi_asymp(mu, epsilon, Psi, chi)
#     betas1[i], Es1[i] =  combined_beta_E(model, n_theta, n_rhat)
#     print(i)
    
# ax.plot(Es1, betas1, label = 'Varying $\\chi$', marker = 'x')

# ########## ANother track originating in the corner
# a0 = 0.143714
# Psi = a0 + (9*mu)/(4*np.pi * epsilon)
# epsilon = 0.1    
# mu = 0.1

# _, crit_chi, _,_ = determine_critical_chi_asymp(mu,epsilon, Psi)

# chis = np.linspace(0,crit_chi,100)

# betas2 = np.zeros(len(a0s))
# Es2 = np.zeros(len(a0s))

# for i, chi in enumerate(chis):
    
#     model = RLoKi_asymp(mu, epsilon, Psi, chi)
#     betas2[i], Es2[i] =  combined_beta_E(model, n_theta, n_rhat)
#     print(i)
    
# ax.plot(Es2, betas2, label = 'Varying $\\chi$', marker = 'x')

### A track for fixed chi, varying a0

# mu = 0.3
# chi = 1e-5
# epsilon = 0.1

# a0s = np.concatenate((np.linspace(0.01, 0.1, 150), np.linspace(0.1, 0.2, 150), np.linspace(0.2, 30, 150)))

# betas3 = np.zeros(len(a0s))
# Es3 = np.zeros(len(a0s))

# for i, a0 in enumerate(a0s):
    
#     Psi = a0 + (9*mu)/(4*np.pi * epsilon)
    
#     _, crit_chi, _,_ = determine_critical_chi_asymp(mu,epsilon, Psi)
    
#     if (chi<= crit_chi):
        
#         model = RLoKi_asymp(mu, epsilon, Psi, chi)
#         betas3[i], Es3[i] =  combined_beta_E(model, n_theta, n_rhat)
        
#     else:
#         print('Oooh, a gap')
#         betas3[i], Es3[i] =  np.NaN, np.NaN
        
#     print(i)
    
# ax.plot(Es3, betas3, label = f'$\\chi$ = {chi}')#, marker = 'x')
# axins = ax.inset_axes([0.55, 0.55, 0.45, 0.45], xlim=(0.12,0.42 ), ylim=(1.2, 1.35))
# axins.plot(Es_base, betas_base, label = '$\\chi$ = 0')
# axins.plot(Es3, betas3, label = f'$\\chi$ = {chi}')
# ax.legend(loc = 'lower right')
# ax.indicate_inset_zoom(axins, edgecolor="black")

############Checking scaling by an honest calculation


epsilons = np.logspace(np.log10(0.01), np.log10(0.05),20)
Psi = 5
C_Psi = -4.962353926522837
D_Psi = 0.6622717575010029
alpha_c = 9.37875751503006
kappa = 18/(5 * np.exp(Psi)*gamma(5/2)*gammainc(5/2,Psi))

betas = np.zeros(len(epsilons))
Es = np.zeros(len(epsilons))

for i, epsilon in enumerate(epsilons):
    mu_c = (4*np.pi*epsilon/9) * (Psi + epsilon**2 * C_Psi + epsilon**4*np.log(epsilon) * 40 * Psi**4 * kappa)
    betas[i], Es[i] = old_beta_energy(mu_c,epsilon, Psi)
    
fig,ax = plt.subplots(1,1)
ax.plot(epsilons, betas, marker = 'x')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\epsilon$')
ax.set_ylabel('$\\beta$')

regression = linregress(np.log10(epsilons), np.log10(betas))
ax.plot(epsilons, 10**(regression[0]*np.log10(epsilons) + regression[1]), color = 'r')
ax.plot(epsilons, 10**((4/3)*np.log10(epsilons) + regression[1]), color = 'k')
beta_scale = regression[0]

fig,ax = plt.subplots(1,1)
ax.plot(epsilons, Es, marker = 'x')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\epsilon$')
ax.set_ylabel('$\\mathcal{E}$')

regression2 = linregress(np.log10(epsilons), np.log10(Es))
ax.plot(epsilons, 10**(regression2[0]*np.log10(epsilons) + regression2[1]), color = 'r')
ax.plot(epsilons, 10**((8/3)*np.log10(epsilons) + regression2[1]), color = 'k')
E_scale = regression2[0]

########## Regenerating figure 17 in LoKi paper

fig, [ax1,ax2] = plt.subplots(2,1)
axins1 = fig.add_axes([0.6, 0.7, 0.25, 0.15])

#### First add the classical King behaviour

epsilon = 1e-6
mu = 0
a0s = np.concatenate((np.linspace(epsilon**2, 1, 50), np.linspace(1, 25, 200)))

Es = np.zeros(len(a0s))
betas = np.zeros(len(a0s))

for i, a0 in enumerate(a0s):
    
    Psi = a0 + (9*mu)/(4*np.pi*epsilon)
    betas[i], Es[i] = old_beta_energy(mu,epsilon, Psi)
    print(i)

ax1.plot(Es, betas, label = '$\\epsilon$ = 0', color = 'k')#, marker = 'x')

# ### Add polytropic behaviour
ax1.plot(Es, 5.77*Es**0.5, color = 'k', linestyle = '--')

# ### Add loaded lines

epsilons = [0.1, 0.05, 0.01]
mu_epsilon = 3

for i, epsilon in enumerate(epsilons):
    
    if (i ==2):
        a0s = np.concatenate((np.logspace(np.log10(epsilon**2), np.log10(0.11*epsilon), 200),np.linspace(0.11*epsilon, 0.12*epsilon , 500) ,np.linspace(0.12*epsilon, 25, 200)))
    else:
        a0s = np.concatenate((np.logspace(np.log10(epsilon**2), np.log10(0.2), 300), np.linspace(0.2, 25, 300)))
    # a0s = np.logspace(np.log10(8*epsilon**2), np.log10(2*epsilon), 100)
    # a0s = np.linspace(0.11*epsilon,0.12*epsilon , 500)
    mu = mu_epsilon * epsilon
    Es = np.zeros(len(a0s))
    betas = np.zeros(len(a0s))
    
    scaled_Es = np.zeros(len(a0s))
    scaled_betas = np.zeros(len(a0s))

    for j, a0 in enumerate(a0s):
        
        Psi = a0 + (9*mu)/(4*np.pi*epsilon)
        betas[j], Es[j] = old_beta_energy(mu,epsilon, Psi)
        
        scaled_betas[j], scaled_Es[j] = epsilon **(-beta_scale) * betas[j], epsilon ** (-E_scale) * Es[j]
        print(f'({i},{j})')
    ax1.plot(Es, betas, label = f'$\\epsilon$ = {epsilon}')#, marker = 'x')
    axins1.plot(scaled_Es, scaled_betas)
    
ax1.set_xlim(0,2)
ax1.set_ylim(0,1.7)
ax1.set_xlabel('$\\mathcal{E}$')
ax1.set_ylabel('$\\beta$')
ax1.set_title('$\\mu/\\epsilon$ = 3')
ax1.legend(loc='lower left')

axins1.set_ylabel('$\\beta \\epsilon^{-4/3}$') 
axins1.set_xlabel('$\\mathcal{E}\\epsilon^{-8/3}$')       
axins1.set_xlim(0,1000)
axins1.set_ylim(0,0.5)

###### Second panel
Psi = 5
C_Psi = -4.962353926522837
D_Psi = 0.6622717575010029
alpha_c = 9.37875751503006
kappa = 18/(5 * np.exp(Psi)*gamma(5/2)*gammainc(5/2,Psi))

epsilons = [0.1, 0.05, 0.01]
for i, epsilon in enumerate(epsilons):
    
    mu_max = 4*np.pi*epsilon*Psi/9
    mu_c = (4*np.pi*epsilon/9) * (Psi + epsilon**2 * C_Psi + epsilon**4*np.log(epsilon) * 40 * Psi**4 * kappa**2 - epsilon**4 * (alpha_c - D_Psi))
    
    critical_model_beta, critical_model_E = old_beta_energy(mu_c,epsilon, Psi)
    
    width = mu_max-mu_c
    
    mus = np.concatenate((np.linspace(0,mu_c-width/2,500) ,np.linspace(mu_c - width/2, mu_c+width/2, 500),np.linspace(mu_c+width/2, mu_c+width, 100)))
    Es = np.zeros(len(mus))
    betas = np.zeros(len(mus))
    
    scaled_Es = np.zeros(len(mus))
    scaled_betas = np.zeros(len(mus))
    
    for j, mu in enumerate(mus):
        
        betas[j], Es[j] = old_beta_energy(mu,epsilon, Psi)
        scaled_betas[j], scaled_Es[j] = epsilon **(-4/3) * betas[j], epsilon ** (-8/3) * Es[j]
        
    ax2.plot(Es, betas, label = f'$\\epsilon$ = {epsilon}')
    ax2.scatter(critical_model_E, critical_model_beta, marker = 'x', color = 'r')
    
ax2.set_xlabel('$\\mathcal{E}$')
ax2.set_ylabel('$\\beta$')
ax2.set_title(f"$\\Psi =$ {Psi}")
ax2.set_xlim(0,1.5)
ax2.set_ylim(0,2)
classical_model_beta, classical_model_E = old_beta_energy(0,1e-6, Psi)
ax2.scatter(classical_model_E, classical_model_beta, marker = 'x', color = 'k', label = 'Classical model')
ax2.legend(loc = 'lower left')
#plt.savefig('fig7_remade')

############# Checking beta/E slope with chi
     
# Psi = 4
# epsilon = 0.1    
# mu = 0.1

# ###### Calculatring the O(1) quantities
# LoKi_model = LoKi(mu, epsilon, Psi)

# M_0 = LoKi_model.M_hat * LoKi_model.density(Psi)
# U_0 = LoKi_model.U_hat * (9 * LoKi_model.density(Psi))/(4*np.pi) ## factors of rh0(Psi) here due to the difference in normalisation employed between the older LoKi and new RLoki quantities.
# rt = LoKi_model.rt

# model = RLoKi_asymp(mu, epsilon, Psi, 0)
# M_0 = asymp_M_hat(model)
# U_0 = W_hat_naive2(model, 100, 1000)

# # E_0 = ((-8 * np.pi**2 * LoKi_model.density(Psi)**2)/(9 * 3**(8/3)) ) * U_0 / M_0**(7/3)
# # beta_0 = ((81 * 3**(2/3))/(16*np.pi**2) ) * (LoKi_model.M_hat**(4/3)) / LoKi_model.density(Psi)**(2/3)
# beta_0, E_0 = combined_beta_E(model, 100, 1000)

# #### Calculating the radially dependent functions required for the O(chi) calculation.
# non_rotating_model = RLoKi_asymp(mu, epsilon, Psi, 0)
# r = non_rotating_model.rhat

# rt_mask =  r < rt # Only integrate between 0 and rt

# r = r[rt_mask]
# psi_0 = non_rotating_model.u_0[rt_mask] + (9*mu)/(4*np.pi * r)
# R1 = np.array(list(map(non_rotating_model.R1, psi_0)))
# rho_0 = np.array(list(map(LoKi_model.density, psi_0)))
# u_10 = non_rotating_model.u_10[rt_mask]

# #### O(chi) correction for beta
# M_1 = -4 * np.pi * LoKi_model.density(Psi) * non_rotating_model.lambda_1 /9
# beta_1 = (81 * 3**(2/3) * 4 * M_0**(1/3) * M_1 ) / (16 * np.pi**2 * LoKi_model.density(Psi)**2 * 3)

# #### O(chi) correction for U

# integrand1 = LoKi_model.density(Psi)/9 * R1 * psi_0 * u_10 * r**2
# integrand2 = rho_0 * u_10 * r**2
# integrand3 = 3 * r**4 * rho_0
# integrand4 = (9 * mu * LoKi_model.density(Psi) * R1 * u_10 * r) / (4 * np.pi * 9)
# integral = simpson(y = integrand1 + integrand2 + integrand3 - integrand4, x = r)

# U_1 = 0.5*(non_rotating_model.alpha_0 * M_1 + non_rotating_model.alpha_1 * M_0) - 2 * np.pi * integral
# E_1 = ((-8 * np.pi**2 * LoKi_model.density(Psi)**2)/(9 * 3**(8/3)) ) * ((3*U_1*M_0 - 7*U_0*M_1) / (3 * M_0 **(10/3)))


# n_theta = 100
# n_rhat = 1000

# _, crit_chi, _,_ = determine_critical_chi_asymp(mu,epsilon, Psi)

# chis = np.linspace(0,crit_chi,100)

# betas = np.zeros(len(chis))
# Es = np.zeros(len(chis))
# Us = np.zeros(len(chis))
# Ms = np.zeros(len(chis))

# for i, chi in enumerate(chis):
    
#     model = RLoKi_asymp(mu, epsilon, Psi, chi)
#     betas[i], Es[i] =  combined_beta_E(model, n_theta, n_rhat)
#     Ms[i] = asymp_M_hat(model)
#     Us[i] = W_hat_naive2(model, n_theta, n_rhat)
#     print(i)
    
# approx_betas = beta_0 + chis * beta_1
# approx_Es = E_0 + chis * E_1
# approx_Us = U_0 + chis * U_1
# approx_Ms = M_0 + chis * M_1

# fig,ax = plt.subplots(1,1)
# ax.plot(chis, betas, label = 'Numerical')
# ax.plot(chis, approx_betas, label = 'Asymptotic approximation')
# ax.set_xlabel('$\\chi$')
# ax.set_ylabel('$\\beta$')
# ax.legend(loc = 'best')

# fig,ax = plt.subplots(1,1)
# ax.plot(chis, Es, label = 'Numerical')
# ax.plot(chis, approx_Es, label = 'Asymptotic approximation')
# ax.set_xlabel('$\\chi$')
# ax.set_ylabel('$\\mathcal{E}$')    
# ax.legend(loc = 'best')

# fig,ax = plt.subplots(1,1)
# ax.plot(chis, Ms, label = 'Numerical')
# ax.plot(chis, approx_Ms, label = 'Asymptotic approximation')
# ax.set_xlabel('$\\chi$')
# ax.set_ylabel('$\\hat{M}$')    
# ax.legend(loc = 'best') 

# fig,ax = plt.subplots(1,1)
# ax.plot(chis, Us, label = 'Numerical')
# ax.plot(chis, approx_Us, label = 'Asymptotic approximation')
# ax.set_xlabel('$\\chi$')
# ax.set_ylabel('$\\hat{U}$')    
# ax.legend(loc = 'best') 
 

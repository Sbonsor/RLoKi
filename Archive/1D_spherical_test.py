#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:34:23 2023

@author: s1984454
"""
from LoKi import LoKi
import numpy as np
from numpy.polynomial.legendre import legfit,legval
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import flt
from scipy.special import gammainc,gamma
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

def rho_hat(psi):
    
    density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
    density = np.nan_to_num(density,copy = False)
        
    return density

def initialise(Psi,epsilon,mu):
    base_model = LoKi(0, epsilon, Psi, pot_only = True)

    rt = base_model.rhat[-1]

    inner_grid = base_model.rhat
    spacing = base_model.rhat[-1] - base_model.rhat[-2]
    outer_grid = np.arange(rt + spacing, 2*rt, spacing)

    r_grid = np.concatenate((inner_grid,outer_grid))

    psi_n1 = np.interp(r_grid, base_model.rhat, base_model.psi)
    u_n1 = psi_n1 - (9 * mu)/(4*np.pi*r_grid)
    R_n1 = rt
    
    return u_n1, psi_n1, R_n1, r_grid

def single_iteration(Psi, mu, epsilon, r_grid, psi_n1, R_n1):

    integrand1 = r_grid * rho_hat(psi_n1)
    integrand2 = r_grid**2 * rho_hat(psi_n1)
    
    integral1 = np.zeros(len(r_grid))
    integral2 = np.zeros(len(r_grid))
    
    for i in range(len(r_grid)):
        integral1[i] =  simps(y = integrand1[0:i+1], x = r_grid[0:i+1])
        integral2[i] =  simps(y = integrand2[0:i+1], x = r_grid[0:i+1])
        
    up = -(9/rho_hat(Psi)) * (integral1 - (1/r_grid)*integral2)
    
    up_deriv = -(9/rho_hat(Psi)) * (1/r_grid**2) * integral2 
    
    A_0 = Psi - (9*mu)/(4*np.pi*epsilon)
    B_0 = 0
    # beta_0 = -R_n1**2 * np.interp(R_n1,r_grid,up_deriv)
    # alpha_0 = A_0 + np.interp(R_n1, r_grid, up) - beta_0/R_n1
    
    u_n = A_0 + B_0/r_grid + up
    psi_n = u_n + (9 * mu)/(4*np.pi*r_grid)

    # interior_grid = r_grid[r_grid<R_n1]
    # exterior_grid = r_grid[r_grid>=R_n1]
    
    # interior_u = A_0 + B_0/interior_grid + up[r_grid<R_n1]
    # exterior_u = alpha_0 + beta_0/exterior_grid
    
    # interior_psi = interior_u + (9 * mu)/(4*np.pi*interior_grid)
    # exterior_psi = exterior_u + (9 * mu)/(4*np.pi*exterior_grid)
    
    # psi_n = np.concatenate((interior_psi,exterior_psi))
    # u_n = psi_n - (9 * mu)/(4*np.pi*r_grid)
    
    # idx_array = np.where(psi_n<0)[0]
    
    # if(len(idx_array) == 0):
    #     R_n = r_grid[-1]
    #     print('Warning: No zero-crossing detected')
    # else:
    #     idx = idx_array[0]
    #     R_n = np.interp(0, np.flip(psi_n[0:idx]), np.flip(r_grid[0:idx]))
        
    return u_n, psi_n, up_deriv, up

def stopping_condition(psi_n1, psi_n, tolerance):
    
    criterion = np.nan_to_num(np.max(abs((psi_n - psi_n1)/psi_n)))

    if criterion < tolerance:
        
        return True
    
    else:
        return False

def run_iteration_scheme(Psi, epsilon, mu, tol, max_iters):
    
    Rs = []
    psis = []
    us = []

    u_n1, psi_n1, R_n1, r_grid = initialise(Psi,epsilon,mu)

   # Rs.append(R_n1)
    psis.append(psi_n1)
    us.append(u_n1)

    ### Iteration
    converged = False
    iteration = 0

    while((converged == False) and (iteration < max_iters)):
        
        u_n, psi_n, up_deriv,up = single_iteration(Psi, mu, epsilon, r_grid, psi_n1, R_n1)
        
        converged = stopping_condition(psi_n1, psi_n, tol)
        
        u_n1 = u_n
        psi_n1 = psi_n
        #R_n1 = R_n
        
        iteration += 1
        
        #Rs.append(R_n1)
        psis.append(psi_n1)
        us.append(u_n1)
        print(iteration)
    
    return Rs, psis, us, r_grid, up_deriv, up

Psi = 5
epsilon = 0.1
mu = 0.1

tol = 1e-3
max_iters = 20

Rs, psis, us, r_grid, up_deriv, up = run_iteration_scheme(Psi, epsilon, mu, tol, max_iters)

idx_array = np.where(psis[-1]<0)[0]

if(len(idx_array) == 0):
    R = r_grid[-1]
    print('Warning: No zero-crossing detected')
else:
    idx = idx_array[0]
    R = np.interp(0, np.flip(psis[-1][0:idx]), np.flip(r_grid[0:idx]))

beta_0 = -R**2 * np.interp(R,r_grid,up_deriv)
alpha_0 = (Psi-(9*mu)/(4*np.pi*epsilon)) + np.interp(R, r_grid, up) - beta_0/R


##### Checking scheme 

ref_model = LoKi(mu, epsilon, Psi)

fig,ax = plt.subplots(1,1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(r_grid,psis[-1], label = 'Iteration scheme')
ax.plot(ref_model.rhat, ref_model.psi, label = 'LoKi model')
ax.set_ylabel('$\\psi$')
ax.set_xlabel('$\\hat{r}$')
ax.set_title('$(\\Psi,\\epsilon,\\mu)=$'+'('+str(Psi) +','+str(epsilon)+','+str(mu)+')')
ax.legend()
    
# psi_n1 = psis[0]
# R_n1 = Rs[0]
# u_n1 = us[0]

# integrand1 = r_grid * rho_hat(psi_n1)
# integrand2 = r_grid**2 * rho_hat(psi_n1)

# integral1 = np.zeros(len(r_grid))
# integral2 = np.zeros(len(r_grid))

# for i in range(len(r_grid)):
#     integral1[i] =  simps(y = integrand1[0:i+1], x = r_grid[0:i+1])
#     integral2[i] =  simps(y = integrand2[0:i+1], x = r_grid[0:i+1])
    
# up = -(9/rho_hat(Psi)) * (integral1 - (1/r_grid)*integral2)

# up_deriv = -(9/rho_hat(Psi)) * (1/r_grid**2) * integral2 

# A_0 = Psi - (9*mu)/(4*np.pi*epsilon)
# B_0 = 0
# # beta_0 = -R_n1**2 * np.interp(R_n1,r_grid,up_deriv)
# # alpha_0 = A_0 + np.interp(R_n1, r_grid, up) - beta_0/R_n1

# u_interior = A_0 + B_0/r_grid + up
# psi_interior = u_interior + (9 * mu)/(4*np.pi*r_grid)


# interior_grid = r_grid#[r_grid<R_n1]
# exterior_grid = r_grid#[r_grid>=R_n1]

# interior_u = A_0 + B_0/interior_grid + up#[r_grid<R_n1]
# exterior_u = alpha_0 + beta_0/exterior_grid

# interior_psi = interior_u + (9 * mu)/(4*np.pi*interior_grid)
# exterior_psi = exterior_u + (9 * mu)/(4*np.pi*exterior_grid)

# psi_n = np.concatenate((interior_psi,exterior_psi))
# u_n = psi_n1 - (9 * mu)/(4*np.pi*r_grid)

# idx_array = np.where(psi_n<0)[0]

# if(len(idx_array) == 0):
#     R_n = r_grid[-1]
#     print('Warning: No zero-crossing in the equatorial plane')
# else:
#     idx = idx_array[0]
#     R_n = np.interp(0, np.flip(psi_n[0:idx]), np.flip(r_grid[0:idx]))


























#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:03:22 2023

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

def legendre_decomp(u,threshold):
    
    coeffs = np.zeros(np.shape(u))
    
    for i in range(np.shape(u)[1]):
        coeffs[:,i] = flt.dlt(u[:,i], closed = True)
    
    coeffs[abs(coeffs) < threshold] = 0
    
    return coeffs

def initialise(Psi, epsilon, mu, chi, lmax):
    
    theta_grid = flt.theta(lmax+1, closed = True)
    
    base_model = LoKi(mu, epsilon, Psi, pot_only = True)
    
    rt = base_model.rhat[-1]
    
    inner_grid = base_model.rhat
    spacing = base_model.rhat[-1] - base_model.rhat[-2]
    outer_grid = np.arange(rt + spacing, 2*rt, spacing)
    
    r_grid = np.concatenate((inner_grid,outer_grid))
    
    psi = np.interp(r_grid, base_model.rhat, base_model.psi)
    psi_theta_r = np.tile(psi,(lmax+1,1))
    
    
    u_theta_r = psi_to_u(psi_theta_r, r_grid, theta_grid, mu, chi)

    return u_theta_r, theta_grid, r_grid, psi_theta_r

def rho_hat(psi):
    
    density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
    density = np.nan_to_num(density,copy = False)
        
    return density

def u_to_psi(u, r_grid, theta_grid, mu, chi):
    
    R,Theta = np.meshgrid(r_grid,theta_grid)
    
    psi = u + 9*mu/(4*np.pi*R) + (9/2)* chi * R**2 * np.sin(Theta)**2
    
    return psi

def psi_to_u(psi, r_grid, theta_grid, mu, chi):
    
    R,Theta = np.meshgrid(r_grid,theta_grid)
    
    u = psi - 9*mu/(4*np.pi*R) - (9/2)* chi * R**2 * np.sin(Theta)**2
    
    return u

def r_power_matrix(power_start, power_end, r_grid, lmax):
    
    r_power_matrix = np.zeros((lmax+1, len(r_grid)))
    powers = np.linspace(power_start,power_end, lmax+1)
    for i in range(lmax+1):
        r_power_matrix[i,:] = np.power(r_grid, powers[i])
        
    return  r_power_matrix

def integrate_matrix(integrand, r_grid):
    
    result = np.zeros(np.shape(integrand))
    
    for i in range(np.shape(integrand)[1]):
        result[:,i] =  simps(y = integrand[:,0:i+1], x = r_grid[0:i+1])
    
    return result

def coefficient_matrix(lmax, r_grid, Psi):
    
    coeffs = np.zeros((lmax+1, len(r_grid)))
    
    for l in range(lmax+1):
        
        coeffs[l,:] = 1/(2*l +1)
    
    coeffs *= -9  / rho_hat(Psi)
    
    return coeffs

def sum_f_l_r(f_l_r, theta_grid,r_grid):
    
    result = np.zeros((len(theta_grid),len(r_grid)))

    for i in range(len(r_grid)):    
        result[:,i] = legval(x = np.cos(theta_grid), c = f_l_r[:,i])
    
    return result

def stopping_condition(psi_n, psi_n1, tolerance):
    
    criterion = np.nan_to_num(np.max(abs((psi_n - psi_n1)/psi_n)))

    if criterion < tolerance:
        
        return True
    
    else:
        return False

def single_iteration(Psi, epsilon, mu, chi, lmax, psi_theta_r_n1, u_theta_r_n1, r_grid, theta_grid, legendre_threshold):
    
    rho_theta_r_n1 = rho_hat(psi_theta_r_n1)

    rho_hat_n1 = legendre_decomp(rho_theta_r_n1,legendre_threshold)

    coeff_matrix = coefficient_matrix(lmax, r_grid, Psi)

    ######Should really pre-calculate these rather than redoing them at every step.
    r1l_matrix = r_power_matrix(1, 1-lmax, r_grid, lmax)
    rl2_matrix = r_power_matrix(2, 2+lmax, r_grid, lmax)
    r_matrix = np.tile(r_grid,(lmax + 1,1))

    rl_matrix = r_matrix / r1l_matrix
    r_l1_matrix = r_matrix / rl2_matrix

    integrand1 = r1l_matrix * rho_hat_n1
    integrand2 = rl2_matrix * rho_hat_n1

    integral1 = coeff_matrix * integrate_matrix(integrand1, r_grid)
    integral2 = coeff_matrix * integrate_matrix(integrand2, r_grid)

    Al = -integral1[:,-1]
    Al[0] = Psi - (9*mu)/(4*np.pi*epsilon)
    Al_matrix = np.tile(np.reshape(Al,(len(Al),1)), (1,len(r_grid)))

    particular_coefficients = integral1 * rl_matrix - integral2 * r_l1_matrix
    particular_solution = sum_f_l_r(particular_coefficients, theta_grid,r_grid)

    homogeneous_coefficients = Al_matrix * rl_matrix 
    homogeneous_solution = sum_f_l_r(homogeneous_coefficients, theta_grid,r_grid)

    u_theta_r_n = homogeneous_solution + particular_solution
    psi_theta_r_n = u_to_psi(u_theta_r_n, r_grid, theta_grid, mu, chi)
    
    return u_theta_r_n, psi_theta_r_n, Al

def run_iteration_scheme(Psi, epsilon, mu, chi, lmax, legendre_threshold, max_iters, tol):
    
    u_theta_r_n1, theta_grid, r_grid, psi_theta_r_n1 = initialise(Psi, epsilon, mu, chi, lmax)
    
    psis = []
    Als = []
    us = []
    
    psis.append(psi_theta_r_n1)
    us.append(u_theta_r_n1)
    
    converged = False
    iteration = 0
    
    while((converged == False) and (iteration < max_iters)):
        
        u_theta_r_n, psi_theta_r_n, Al = single_iteration(Psi, epsilon, mu, chi, lmax, psi_theta_r_n1, u_theta_r_n1, r_grid, theta_grid, legendre_threshold)
        
        psis.append(psi_theta_r_n)
        Als.append(Al)
        us.append(u_theta_r_n)
        
        converged = stopping_condition(psi_theta_r_n, psi_theta_r_n1, tol)
        
        iteration += 1
        
        print(iteration)
        
        if(converged == True):
            print('Solution converged!')
        
        if(iteration == max_iters):
            print('Solution not converged in specified number of iterations.')
            
        u_theta_r_n1 = u_theta_r_n
        psi_theta_r_n1 = psi_theta_r_n
        
    
    return psis, us, Als, r_grid, theta_grid

def interp_equatorial_plane(psi_theta_r, theta_grid, r_grid):
    
    interpolator = RegularGridInterpolator((theta_grid, r_grid), psi_theta_r)

    psi_equatorial =  interpolator((np.pi/2,r_grid))
    
    return psi_equatorial

### Inputs: True parameters

# Psi = 5
# chi = 0
# epsilon = 0.1
# mu = 0.1

# ### Inputs: Parameters required for numerics
# lmax = 10
# legendre_threshold = 1e-12
# max_iters = 40
# tol = 1e-3

# psis, us, Als, r_grid, theta_grid = run_iteration_scheme(Psi, epsilon, mu, chi, lmax, legendre_threshold, max_iters, tol)


########### Checking chi = 0 case

Psi = 5
chi = 0
epsilon = 0.01
mu = 0

### Inputs: Parameters required for numerics
lmax = 10
legendre_threshold = 1e-12
max_iters = 40
tol = 1e-3


psis, us, Als, r_grid, theta_grid = run_iteration_scheme(Psi, epsilon, mu, chi, lmax, legendre_threshold, max_iters, tol)

base_model = LoKi(mu, epsilon, Psi, pot_only = True)

fig1,ax1 = plt.subplots(1,1)
ax1.axhline(y=0, linewidth = 0.5, color = 'k', linestyle = '--')
ax1.set_xlabel('$\\hat{r}$')
ax1.set_ylabel('$\\psi(\\hat{r}$')
ax1.set_title('$(\\Psi,\\chi,\\epsilon,\\mu)=$'+'('+str(Psi)+','+str(chi)+','+str(epsilon)+','+str(mu)+')')
ax1.plot(base_model.rhat, base_model.psi, label = 'LoKi model')
ax1.plot(r_grid,psis[-1][5,:], label = 'Iteration scheme')
ax1.legend()

########## Trying to reproduce fig.2 in AL 2012
# Psis = np.linspace(1,6,6)
# chis = [3e-3, 1.6e-3, 1e-3, 3.5e-4, 2e-4, 5e-5]
# lmax = 10
# mu = 0.1
# epsilon = 0.1
# legendre_threshold = 5e-12
# max_iters = 40
# tol = 1e-3


# fig2,ax2 = plt.subplots(1,1)

# for i in range(6):
#     print(Psis[i])
#     psis, us, Als, r_grid, theta_grid = run_iteration_scheme(Psis[i], epsilon, mu, chis[i], lmax, legendre_threshold, max_iters, tol)
    
#     density = rho_hat(psis[-1])

#     ax2.plot(r_grid,density[0,:]/rho_hat(Psis[i]), label = '$\\theta =$ ' + str(theta_grid[0]), color = 'k', linestyle = '--', linewidth = 0.5)
#     ax2.plot(r_grid,density[6,:]/rho_hat(Psis[i]), label = '$\\theta =$ ' + str(theta_grid[6]), color = 'k', linewidth = 0.5)


# ax2.set_yscale('log')
# ax2.set_xscale('log')
# ax2.set_xlim(0.1,300)
# ax2.set_ylim(1e-8,1)
# #ax2.legend()
# ax2.set_ylabel('$\\rho/\\rho_0$')
# ax2.set_xlabel('$\\hat{r}$')


# equatorial = interp_equatorial_plane(psis[-1], theta_grid, r_grid)
########### Development hell


### Calculate base case development

#u_theta_r_n1, theta_grid, r_grid, psi_theta_r_n1 = initialise(Psi, epsilon, mu, lmax)

### RUn single iter

# u_theta_r_n, psi_theta_r_n, Al = single_iteration(Psi, epsilon, mu, chi, lmax, psi_theta_r_n1, u_theta_r_n1, r_grid, legendre_threshold)


# ### Single iteration development, use to diagnose issues.
# idx = 4
# legendre_threshold = 1e-16

# Psi = Psis[idx]
# psi_theta_r_n1 = psis[idx]
# chi = chis[idx]

# rho_theta_r_n1 = rho_hat(psi_theta_r_n1)

# rho_hat_n1 = legendre_decomp(rho_theta_r_n1,legendre_threshold)

# coeff_matrix = coefficient_matrix(lmax, r_grid, Psi)

# ######Should really pre-calculate these rather than redoing them at every step.
# r1l_matrix = r_power_matrix(1, 1-lmax, r_grid, lmax)
# rl2_matrix = r_power_matrix(2, 2+lmax, r_grid, lmax)
# r_matrix = np.tile(r_grid,(lmax + 1,1))

# rl_matrix = r_matrix / r1l_matrix
# r_l1_matrix = r_matrix / rl2_matrix

# integrand1 = r1l_matrix * rho_hat_n1
# integrand2 = rl2_matrix * rho_hat_n1

# integral1 = coeff_matrix * integrate_matrix(integrand1, r_grid)
# integral2 = coeff_matrix * integrate_matrix(integrand2, r_grid)

# Al = -integral1[:,-1]
# Al[0] = Psi - (9*mu)/(4*np.pi*epsilon)
# Al_matrix = np.tile(np.reshape(Al,(len(Al),1)), (1,len(r_grid)))

# particular_coefficients = integral1 * rl_matrix - integral2 * r_l1_matrix
# particular_solution = sum_f_l_r(particular_coefficients, theta_grid,r_grid)

# homogeneous_coefficients = Al_matrix * rl_matrix 
# homogeneous_solution = sum_f_l_r(homogeneous_coefficients, theta_grid,r_grid)

# u_theta_r_n = homogeneous_solution + particular_solution
# psi_theta_r_n = u_to_psi(u_theta_r_n, r_grid, theta_grid, mu, chi)













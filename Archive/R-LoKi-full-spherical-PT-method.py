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

def initialise(Psi, epsilon, mu, lmax):
    
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

    return u_theta_r, theta_grid, r_grid, rt, psi_theta_r

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

def interpolate_to_point(xnew, ynew, x, y, z):
    
    f = RegularGridInterpolator((y, x), z)
    
    return f((ynew, xnew))

def find_R_n(r_grid, theta_grid, psi_theta_r_n):

    psi = RegularGridInterpolator((theta_grid, r_grid), psi_theta_r_n)
    psi_equatorial = psi((np.pi/2,r_grid))
    
    idx_array = np.where(psi_equatorial<0)[0]
    
    if(len(idx_array) == 0):
        R_n = r_grid[-1]
        print('Warning: No zero-crossing in the equatorial plane')
    else:
        idx = idx_array[0]
        R_n = np.interp(0, np.flip(psi_equatorial[0:idx]), np.flip(r_grid[0:idx]))
    
    return R_n, psi_equatorial

def single_iteration(Psi, epsilon, mu, lmax, chi, u_theta_r_n1, theta_grid, r_grid, R_n1, legendre_threshold):
    
    psi_theta_r_n1 =  u_to_psi(u_theta_r_n1, r_grid, theta_grid, mu, chi)

    rho_theta_r_n1 = rho_hat(psi_theta_r_n1)

    rho_hat_n1 = legendre_decomp(rho_theta_r_n1,legendre_threshold)

    lambda_n1 = coefficient_matrix(lmax, r_grid, Psi)

    r1l_matrix = r_power_matrix(1, 1-lmax, r_grid, lmax)
    rl2_matrix = r_power_matrix(2, 2+lmax, r_grid, lmax)
    r_matrix = np.tile(r_grid,(lmax + 1,1))

    rl_matrix = r_matrix / r1l_matrix
    r_l1_matrix = r_matrix / rl2_matrix

    integrand1 = r1l_matrix * rho_hat_n1
    integrand2 = rl2_matrix * rho_hat_n1

    integral1 = lambda_n1 * integrate_matrix(integrand1, r_grid)
    integral2 = lambda_n1 * integrate_matrix(integrand2, r_grid)

    Al = -integral1[:,-1]
    Bl = Al * -epsilon**np.linspace(1,2*lmax+1,lmax+1)

    Al[0] = 0
    Bl[0] = 0

    particular_coefficients = integral1 * rl_matrix - integral2 * r_l1_matrix
    particular_solution = sum_f_l_r(particular_coefficients, theta_grid,r_grid)

    up = interpolate_to_point(R_n1, np.pi/2, r_grid, theta_grid, particular_solution) 

    uh_coeffs = Al * (R_n1 ** np.linspace(0, lmax, lmax+1)) + Bl * (R_n1 ** np.linspace(-1,-lmax-1, lmax+1))
    uh = legval(x = np.cos(np.pi/2), c = uh_coeffs)

    Bl[0] = (Psi - (9 * mu / (4 * np.pi * epsilon)) + (9 * mu /(4 * np.pi * R_n1)) + (9/2) * chi * R_n1**2 + uh + up) / (1/epsilon - 1/R_n1)
    Al[0] = Psi -  (9 * mu / (4 * np.pi * epsilon)) - Bl[0]/epsilon

    Al_matrix = np.tile(np.reshape(Al,(len(Al),1)), (1,len(r_grid)))
    Bl_matrix = np.tile(np.reshape(Bl,(len(Bl),1)), (1,len(r_grid)))

    homogeneous_coefficients = Al_matrix * rl_matrix + Bl_matrix * r_l1_matrix
    homogeneous_solution = sum_f_l_r(homogeneous_coefficients, theta_grid,r_grid)

    u_theta_r_n = homogeneous_solution + particular_solution
    psi_theta_r_n = u_to_psi(u_theta_r_n, r_grid, theta_grid, mu, chi)

    R_n, psi_equatorial = find_R_n(r_grid, theta_grid, psi_theta_r_n)
    
    return u_theta_r_n, R_n, psi_theta_r_n, Al, Bl, psi_equatorial

def stopping_condition(psi_n, psi_n1, tolerance):
    
    criterion = np.nan_to_num(np.max(abs((psi_n - psi_n1)/psi_n)))

    if criterion < tolerance:
        
        return True
    
    else:
        return False

def calculate_psi(Psi, chi, epsilon, mu, lmax, legendre_threshold, max_iters, tol):
    
    u_theta_r_n1, theta_grid, r_grid, R_n1, psi_theta_r_n1 = initialise(Psi, epsilon, mu, lmax)
    
    psis = []
    us = []
    Rs = []
    Als = []
    Bls = []
    psi_equatorials = []

    psis.append(psi_theta_r_n1)
    Rs.append(R_n1)
    us.append(u_theta_r_n1)
    psi_equatorials.append(psi_theta_r_n1[0,:])
    

    iteration = 0
    converged = False

    while((converged == False) and (iteration < max_iters)):
        
        u_theta_r_n, R_n, psi_theta_r_n, Al, Bl,psi_equatorial = single_iteration(Psi, epsilon, mu, lmax, chi, u_theta_r_n1, theta_grid, r_grid, R_n1, legendre_threshold)
        
        converged = stopping_condition(psi_theta_r_n, psi_theta_r_n1, tol)
        
        iteration += 1
        print(iteration)
        if(iteration == max_iters):
            print('Solution not converged in specified number of iterations.')
            
        u_theta_r_n1 = u_theta_r_n
        psi_theta_r_n1 = psi_theta_r_n
        R_n1 = R_n
        
        psis.append(psi_theta_r_n)
        Rs.append(R_n)
        Als.append(Al)
        Bls.append(Bl)
        psi_equatorials.append(psi_equatorial)
        us.append(u_theta_r_n)
    
    if(iteration < max_iters):
        print('Converged.')
    
    return psis, r_grid, theta_grid, Als, Bls, Rs, psi_equatorials, us

### Inputs: True parameters

Psi = 5
chi = 1e-4
epsilon = 0.01
mu = 0

### Inputs: Parameters required for numerics
lmax = 10
legendre_threshold = 1e-12
max_iters = 5
tol = 1e-3

psis, r_grid, theta_grid, Als, Bls, Rs, psi_equatorials, us = calculate_psi(Psi, chi, epsilon, mu, lmax, legendre_threshold, max_iters, tol)


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']       
                      
fig1,ax1 = plt.subplots(1,1)
for i in range(max_iters+1):
    ax1.plot(r_grid,psi_equatorials[i])
    
    if (i>0):
        ax1.axvline(x = Rs[i-1], color = colors[i])
ax1.axhline(y=0, linewidth = 0.5, color = 'k', linestyle = '--')
ax1.set_xlabel('$\\hat{r}$')
ax1.set_ylabel('$\\psi(\\hat{r},\\pi/2)$')
ax1.set_title('$(\\Psi,\\chi,\\epsilon,\\mu)=$'+'('+str(Psi)+','+str(chi)+','+str(epsilon)+','+str(mu)+')')

fig3,ax3 = plt.subplots(1,1)
for i in range(max_iters+1):
    ax3.plot(r_grid,us[i][5,:])
ax3.set_xlabel('$\\hat{r}$')
ax3.set_ylabel('$u(\\hat{r},\\pi/2)$')
ax3.set_title('$(\\Psi,\\chi,\\epsilon,\\mu)=$'+'('+str(Psi)+','+str(chi)+','+str(epsilon)+','+str(mu)+')')

### Checking R_n determination

# psi = RegularGridInterpolator((theta_grid, r_grid), psis[5])
# psi_equatorial = psi((np.pi/2,r_grid))

# idx_array = np.where(psi_equatorial<0)[0]

# if(len(idx_array) == 0):
#     R_n = r_grid[-1]
#     print('Warning: No zero-crossing in the equatorial plane')
# else:
#     idx = idx_array[0]
#     R_n = np.interp(0, np.flip(psi_equatorial[0:idx]), np.flip(r_grid[0:idx]))

# fig2, ax2 = plt.subplots(1,1)
# ax2.axhline(y=0)
# ax2.plot(r_grid,psi_equatorial)
# ax2.axvline(x=R_n)


### Code for iteration process development

# ### Initialise state
# u_theta_r_n1, theta_grid, r_grid, R_n1, psi_theta_r_n1 = initialise(Psi, epsilon, mu, lmax)

# iteration = 0
# converged = False

# while((converged == False) and (iteration < max_iters)):
    
#     u_theta_r_n, R_n, psi_theta_r_n, Al, Bl = single_iteration(Psi, epsilon, mu, lmax, chi, u_theta_r_n1, theta_grid, r_grid, R_n1, legendre_threshold)
    
#     converged = stopping_condition(psi_theta_r_n, psi_theta_r_n1, tol)
    
#     iteration += 1
#     print(iteration)
#     if(iteration == max_iters):
#         print('Solution not converged in specified number of iterations.')
        
#     u_theta_r_n1 = u_theta_r_n
#     psi_theta_r_n1 = psi_theta_r_n
    

### Calculate base case development

# theta_grid = flt.theta(lmax+1, closed = True)

# base_model = LoKi(mu, epsilon, Psi, pot_only = True)

# #r_grid = np.linspace(epsilon, 1.5*base_model.rhat[-1],1000)
# r_grid = base_model.rhat

# psi_0 = np.interp(r_grid, base_model.rhat, base_model.psi)
# psi_theta_r_n1 = np.tile(psi_0,(lmax+1,1))

# R_n1 = base_model.rhat[-1]

# u_theta_r_n1 = psi_to_u(psi_theta_r_n1, r_grid, theta_grid, mu, chi)

# ### Single iteration development

# psi_theta_r_n1 =  u_to_psi(u_theta_r_n1, r_grid, theta_grid, mu, chi)

# rho_theta_r_n1 = rho_hat(psi_theta_r_n1)

# rho_hat_n1 = legendre_decomp(rho_theta_r_n1,legendre_threshold)

# lambda_n1 = coefficient_matrix(lmax, r_grid, Psi)

# r1l_matrix = r_power_matrix(1, 1-lmax, r_grid, lmax)
# rl2_matrix = r_power_matrix(2, 2+lmax, r_grid, lmax)
# r_matrix = np.tile(r_grid,(lmax + 1,1))

# rl_matrix = r_matrix / r1l_matrix
# r_l1_matrix = r_matrix / rl2_matrix

# integrand1 = r1l_matrix * rho_hat_n1
# integrand2 = rl2_matrix * rho_hat_n1

# integral1 = lambda_n1 * integrate_matrix(integrand1, r_grid)
# integral2 = lambda_n1 * integrate_matrix(integrand2, r_grid)

# Al = -integral1[:,-1]
# Bl = Al * -epsilon**np.linspace(1,2*lmax+1,lmax+1)

# Al[0] = 0
# Bl[0] = 0

# particular_coefficients = integral1 * rl_matrix - integral2 * r_l1_matrix
# particular_solution = sum_f_l_r(particular_coefficients, theta_grid,r_grid)

# up = interpolate_to_point(R_n1, np.pi/2, r_grid, theta_grid, particular_solution) 

# uh_coeffs = Al * (R_n1 ** np.linspace(0, lmax, lmax+1)) + Bl * (R_n1 ** np.linspace(-1,-lmax-1, lmax+1))
# uh = legval(x = np.cos(np.pi/2), c = uh_coeffs)

# Bl[0] = (Psi - (9 * mu / (4 * np.pi * epsilon)) + (9 * mu /(4 * np.pi * R_n1)) + (9/2) * chi * R_n1**2 + uh + up) / (1/epsilon - 1/R_n1)
# Al[0] = Psi -  (9 * mu / (4 * np.pi * epsilon)) - Bl[0]/epsilon

# Al_matrix = np.tile(np.reshape(Al,(len(Al),1)), (1,len(r_grid)))
# Bl_matrix = np.tile(np.reshape(Bl,(len(Bl),1)), (1,len(r_grid)))

# homogeneous_coefficients = Al_matrix * rl_matrix + Bl_matrix * r_l1_matrix
# homogeneous_solution = sum_f_l_r(homogeneous_coefficients, theta_grid,r_grid)

# u_theta_r_n = homogeneous_solution + particular_solution
# psi_theta_r_n = u_to_psi(u_theta_r_n, r_grid, theta_grid, mu, chi)

# psi = RegularGridInterpolator((theta_grid, r_grid), psi_theta_r_n)
# psi_equatorial = psi((np.pi/2,r_grid))
# R_n = np.interp(0, psi_equatorial[::-1], r_grid[::-1])

















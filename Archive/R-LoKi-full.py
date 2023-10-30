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

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

def boundary_r_values(theta,Psi,mu,epsilon,chi):
    r_sols = []
    r_sols.append(epsilon)
    n_samp = len(theta)
    
    for i in range(n_samp-2):
        
        a = 9/2 * chi * np.sin(theta[i+1])**2
        b = -9 * mu / (4 * np.pi * epsilon)
        c = 9 * mu / (4 * np.pi ) 

        poly = Polynomial(coef = [c,b,0,a])
        roots = poly.roots()
        
        if len(roots) == 1:
            r_sols.append(roots[0])
        else:
            r_sols.append(roots[1])        
    r_sols.append(epsilon)
    
    return np.array(r_sols)

def legendre_decomp(u,threshold):
    
    coeffs = np.zeros(np.shape(u))
    
    for i in range(np.shape(u)[1]):
        coeffs[:,i] = flt.dlt(u[:,i], closed = True)
    
    coeffs[abs(coeffs) < threshold] = 0
    
    return coeffs

def generate_r_power_matrix(power_start, power_end, r_grid, lmax):
    
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

def generate_coefficient_matrix(lmax,r_grid, Psi):
    
    coeffs = np.zeros((lmax+1, len(r_grid)))
    
    for l in range(lmax+1):
        coeffs[l,:] = 1/(2*l +1)
    
    coeffs *= -9/rho_hat(Psi)
    
    return coeffs
    
def rho_hat(psi):
        
    density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
    density = np.nan_to_num(density,copy = False)
        
    return density

def quadrature_solution(Psi,rho_theta_r_n_minus_1, r_grid,lmax):
    
    rho_hat_1n = legendre_decomp(rho_theta_r_n_minus_1,1e-14)
    
    r1_minus_l_matrix = generate_r_power_matrix(1, 1-lmax, r_grid, lmax)
    r_l_plus_2_matrix = generate_r_power_matrix(2, 2+lmax, r_grid, lmax)
    r_matrix = np.tile(r_grid,(lmax + 1,1))

    r_l_matrix = (1/r1_minus_l_matrix)*r_matrix
    r_minus_l_plus_one = (1/r_l_plus_2_matrix) * r_matrix

    integrand1 = r1_minus_l_matrix * rho_hat_1n
    integrand2 = r_l_plus_2_matrix * rho_hat_1n

    integral1 = integrate_matrix(integrand1, r_grid)
    integral2 = integrate_matrix(integrand2, r_grid)
    
    coeff_matrix = generate_coefficient_matrix(lmax,r_grid, Psi) 
    
    component1 =  coeff_matrix*integral1
    component2 =  coeff_matrix*integral2
    
    A_l = -component1[:,-1]

    solution_with_no_constants = r_l_matrix * component1 - r_minus_l_plus_one * component2
    
    return solution_with_no_constants,A_l, r_l_matrix , r_minus_l_plus_one

def sum_f_l_r(f_l_r, theta_grid,r_grid):
    
    result = np.zeros((len(theta_grid),len(r_grid)))

    for i in range(len(r_grid)):    
        result[:,i] = legval(x = np.cos(theta_grid), c = f_l_r[:,i])
    
    return result

def u_to_psi(u, r_grid, theta_grid, mu, chi):
    
    R,Theta = np.meshgrid(r_grid,theta_grid)
    
    psi = u + 9*mu/(4*np.pi*R) + (9/2)* chi * R**2 * np.sin(Theta)**2
    
    return psi

def psi_to_u(psi, r_grid, theta_grid, mu, chi):
    
    R,Theta = np.meshgrid(r_grid,theta_grid)
    
    u = psi - 9*mu/(4*np.pi*R) - (9/2)* chi * R**2 * np.sin(Theta)**2
    
    return u

def single_iter(u_n1, constants, r_grid, theta_grid, boundary_r):
    
    lmax, epsilon, mu, Psi, chi = constants

    psi_theta_r = u_to_psi(u_n1, r_grid, theta_grid, mu, chi)
    rho_theta_r = rho_hat(psi_theta_r)
    
    f_l_r, A_l, r_l, r_l1 = quadrature_solution(Psi, rho_theta_r, r_grid, lmax)
    F_theta_r = sum_f_l_r(f_l_r, theta_grid, r_grid)
    
    A = np.zeros((lmax+1, lmax+1))
    I = np.identity(lmax+1)
    
    for i in range(lmax+1):
        A[:,i] = boundary_r ** (-i - 1) * legval(x = np.cos(theta_grid), c = I[:,i])
        
    b = np.zeros(lmax+1)

    for i in range(lmax+1):
        rb = boundary_r[i]
        theta = theta_grid[i]
        
        F_eval = np.interp(rb, xp = r_grid, fp = F_theta_r[i,:])
        l_vector = np.linspace(0,lmax, lmax+1)
        b[i] = Psi - 9 * mu/(4 * np.pi * rb) - (9/2) * chi * rb**2 * np.sin(theta)**2 - F_eval - legval(np.cos(theta), c = rb ** l_vector * A_l)

    B_l = np.linalg.solve(A,b)
    
    Al_Bl_recon_coeffs = np.tile(np.reshape(A_l,(lmax+1, 1)), (1, len(r_grid))) * r_l + np.tile(np.reshape(B_l, (lmax+1,1)), (1, len(r_grid))) * r_l1

    Al_Bl_recon = sum_f_l_r(Al_Bl_recon_coeffs, theta_grid, r_grid)

    u_n =  Al_Bl_recon + F_theta_r
    
    return u_n

def initialise_iteration(constants):
    
    lmax, epsilon, mu, Psi, chi = constants
    
    theta_grid = flt.theta(lmax+1, closed = True)
    boundary_r = boundary_r_values(theta_grid, Psi, mu, epsilon, chi)
    
    base_model = LoKi(mu, epsilon, Psi, pot_only = True)
    r_grid = base_model.rhat
    psi = base_model.psi
    
    
    psi_theta_r = np.tile(psi,(lmax+1,1))
    
    u_theta_r = psi_to_u(psi_theta_r, r_grid, theta_grid, mu, chi)
    
    return u_theta_r,r_grid,theta_grid,boundary_r, psi_theta_r

def stopping_condition(psi_n, psi_n1, tolerance):
    
    criterion = np.max(abs((psi_n - psi_n1)/psi_n))

    if criterion < tolerance:
        
        return True
    
    else:
        return False
    
# lmax = 10
# epsilon = 0.1 
# mu = 1e-4
# Psi = 5
# chi = 1e-4

# converged = False
# tolerance = 1e-3

# u_n1, r_grid, theta_grid, boundary_r, psi_n1 = initialise_iteration([lmax, epsilon, mu, Psi, chi])

# u_n = single_iter(u_n1, [lmax, epsilon, mu, Psi, chi], r_grid, theta_grid, boundary_r)


# while(converged == False):
    
#     u_n = single_iter(u_n1, [lmax, epsilon, mu, Psi, chi], r_grid, theta_grid, boundary_r)
#     psi_n = u_to_psi(u_n, r_grid, theta_grid, mu, chi)
    
#     converged = stopping_condition(psi_n, psi_n1, tolerance)
    
#     u_n1 = u_n
#     psi_n1 = psi_n


############### Developing single iteration

lmax = 10
epsilon = 0.1 
mu = 1e-4
Psi = 5
chi = 1e-4
constants = (lmax, epsilon, mu, Psi, chi)

u_theta_r,r_grid,theta_grid,boundary_r, psi_theta_r = initialise_iteration(constants)
rho_theta_r = rho_hat(psi_theta_r)

f_l_r, A_l, r_l, r_l1 = quadrature_solution(Psi,rho_theta_r, r_grid,lmax)
F_theta_r = sum_f_l_r(f_l_r, theta_grid, r_grid)

F_theta_r = sum_f_l_r(f_l_r, theta_grid,r_grid)

A = np.zeros((lmax+1,lmax+1))
I = np.identity(lmax+1)

for i in range(lmax+1):
    A[:,i] = boundary_r ** (-i - 1) * legval(x=np.cos(theta_grid),c = I[:,i])
    
b = np.zeros(lmax+1)

for i in range(lmax+1):
    rb = boundary_r[i]
    theta = theta_grid[i]
    
    F_eval = np.interp(rb, xp = r_grid, fp = F_theta_r[i,:])
    l_vector = np.linspace(0,lmax,lmax+1)
    b[i] = Psi - 9 * mu/(4 * np.pi * rb) - (9/2) * chi * rb**2 * np.sin(theta)**2 - F_eval - legval(np.cos(theta),c = rb ** l_vector * A_l)

B_l = np.linalg.solve(A,b)


# Al_Bl_recon_coeffs = np.tile(np.reshape(A_l,(lmax+1,1)), (1,len(r_grid))) * r_l + np.tile(np.reshape(B_l,(lmax+1,1)), (1,len(r_grid))) * r_l1

# Al_Bl_recon = sum_f_l_r(Al_Bl_recon_coeffs, theta_grid, r_grid)

# u_n =  Al_Bl_recon + F_theta_r



######## Testing quadrature solution

# r_grid = np.linspace(1,10,1000)
# lmax = 2
# n_samp = lmax+1
# theta_grid = flt.theta(n_samp, closed = True)

# rho = np.zeros(len(r_grid))

# rho_hat_1n = np.tile(r_grid, (n_samp,1))

# r1_minus_l_matrix = generate_r_power_matrix(1, 1-lmax, r_grid, lmax)
# r_l_plus_2_matrix = generate_r_power_matrix(2, 2+lmax, r_grid, lmax)
# r_matrix = np.tile(r_grid,(n_samp,1)) 

# integrand1 = r1_minus_l_matrix * rho_hat_1n
# integrand2 = r_l_plus_2_matrix * rho_hat_1n  

# comparison_1 = generate_r_power_matrix(2, 2-lmax, r_grid, lmax)
# comparison_2 = generate_r_power_matrix(3, 3+lmax, r_grid, lmax)              


# l_vector = np.array([0,1,2])
# one = np.ones((n_samp,len(r_grid)))
# l_matrix1 = np.zeros((n_samp,len(r_grid)))
# integration_1 = np.zeros((n_samp,len(r_grid)))

# for i in range(len(r_grid)):
#     l_matrix1[:,i] = 1/(3-l_vector)
#     integration_1[:,i] = r_grid[i]**(3-l_vector)
# result_of_integration1 = l_matrix1 * (integration_1 - one)

# l_vector = np.array([0,1,2])
# one = np.ones((n_samp,len(r_grid)))
# l_matrix2 = np.zeros((n_samp,len(r_grid)))
# integration_2 = np.zeros((n_samp,len(r_grid)))

# for i in range(len(r_grid)):
#     l_matrix2[:,i] = 1/(l_vector+4)
#     integration_2[:,i] = r_grid[i]**(l_vector+4)
# result_of_integration2 = l_matrix2 * (integration_2 - one)
    
# integral1 = integrate_matrix(integrand1, r_grid)
# integral2 = integrate_matrix(integrand2, r_grid)    

# check_coeffs = generate_coefficient_matrix(lmax,r_grid, 2) 
# def generate_rl_matrix(r_grid,lmax):
#     rl_matrix = np.zeros((lmax+1, len(r_grid)))
#     powers = np.linspace(1,1-lmax, lmax+1)
#     for i in range(lmax+1):
#         rl_matrix[i,:] = np.power(r_grid, powers[i])
        
#     return rl_matrix

# def generate_rl1_matrix(r_grid,lmax):
#     rl1_matrix = np.zeros((lmax+1, len(r_grid)))
#     powers = np.linspace(2,2+lmax, lmax+1)
#     for i in range(lmax+1):
#         rl1_matrix[i,:] = np.power(r_grid, powers[i])
        
#     return rl1_matrix    
    




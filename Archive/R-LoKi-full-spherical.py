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
from scipy import linalg
#from sklearn.linear_model import Ridge


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

def quadrature_solution(Psi,rho_theta_r_n_minus_1, r_grid,lmax,threshold):
    
    rho_hat_1n = legendre_decomp(rho_theta_r_n_minus_1,threshold)
    
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
    
    solution_with_no_constants = r_l_matrix * component1 - r_minus_l_plus_one * component2
    
    return solution_with_no_constants, r_l_matrix , r_minus_l_plus_one

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

def single_iter(u_n1, constants, r_grid, theta_grid,threshold):
    
    lmax, epsilon, mu, Psi, chi = constants

    psi_theta_r = u_to_psi(u_n1, r_grid, theta_grid, mu, chi)
    rho_theta_r = rho_hat(psi_theta_r)
    
    f_l_r, r_l, r_l1 = quadrature_solution(Psi, rho_theta_r, r_grid, lmax,threshold)
    F_theta_r = sum_f_l_r(f_l_r, theta_grid, r_grid)
    
    A1 = np.zeros((lmax + 1,2*lmax + 2))
    A2 = np.zeros((lmax + 1,2*lmax + 2))
    
    I = np.identity(lmax+1)
    
    for l in range(lmax+1):
        
        A1[:,l] = l * legval(x = np.cos(theta_grid), c = I[:,l])
        A1[:,lmax + 1 + l] = -(l+1) * legval(x = np.cos(theta_grid), c = I[:,l])
        
        A2[:,l] = legval(x = np.cos(theta_grid), c = I[:,l])
        A2[:,lmax + 1 + l] = legval(x = np.cos(theta_grid), c = I[:,l])
        
    A = np.vstack((A1,A2))    
    b  = np.zeros((2*lmax + 2,1))
    b[lmax+1:, 0] = Psi - (9 * mu)/(4 * np.pi * epsilon)
    #b[lmax+1:, 0] = Psi - (9 * mu)/(4 * np.pi * epsilon) - (9/2) * chi * epsilon**2 * np.sin(theta_grid)**2
    
    coefficients = linalg.solve(A,b)

    A_l_tild = coefficients[:lmax+1]
    B_l_tild = coefficients[lmax+1:]
    
    A_l = A_l_tild / np.reshape(r_l[:,0],(lmax +1 ,1))
    B_l = B_l_tild / np.reshape(r_l1[:,0],(lmax +1 ,1))
    
    Al_Bl_recon_coeffs = np.tile(np.reshape(A_l,(lmax+1, 1)), (1, len(r_grid))) * r_l + np.tile(np.reshape(B_l, (lmax+1,1)), (1, len(r_grid))) * r_l1

    Al_Bl_recon = sum_f_l_r(Al_Bl_recon_coeffs, theta_grid, r_grid)

    u_n =  Al_Bl_recon + F_theta_r
    
    return u_n, A_l.ravel(), B_l.ravel()

def initialise_iteration(constants):
    
    lmax, epsilon, mu, Psi, chi = constants
    
    theta_grid = flt.theta(lmax+1, closed = True)
    
    base_model = LoKi(mu, epsilon, Psi, pot_only = True)
    r_grid = np.linspace(epsilon, 1.5*base_model.rhat[-1],1000)
    psi = np.interp(r_grid, base_model.rhat, base_model.psi)
    
    psi_theta_r = np.tile(psi,(lmax+1,1))
    
    u_theta_r = psi_to_u(psi_theta_r, r_grid, theta_grid, mu, chi)
    
    return u_theta_r,r_grid,theta_grid, psi_theta_r

def stopping_condition(psi_n, psi_n1, tolerance):
    
    criterion = np.max(abs((psi_n - psi_n1)/psi_n))

    if criterion < tolerance:
        
        return True
    
    else:
        return False

def calculate_psi(lmax,epsilon,mu,Psi,chi, legendre_threshold, tolerance):
    
    converged = False
    
    u_n1, r_grid, theta_grid, psi_n1 = initialise_iteration([lmax, epsilon, mu, Psi, chi])

    u_n, Al, Bl = single_iter(u_n1, [lmax, epsilon, mu, Psi, chi], r_grid, theta_grid,legendre_threshold)
    psi_n = u_to_psi(u_n, r_grid, theta_grid, mu, chi)
    converged = stopping_condition(psi_n, psi_n1, tolerance)

    u_n1 = u_n
    psi_n1 = psi_n

    iteration = 1    

    while(converged == False):
        
        u_n, Al, Bl = single_iter(u_n1, [lmax, epsilon, mu, Psi, chi], r_grid, theta_grid,legendre_threshold)
        psi_n = u_to_psi(u_n, r_grid, theta_grid, mu, chi)
        
        converged = stopping_condition(psi_n, psi_n1, tolerance)
        
        u_n1 = u_n
        psi_n1 = psi_n
        
        print(iteration)
        iteration += 1
        
    return psi_n, r_grid, theta_grid

## Test run to prouce fig 4 in Varri & Bertin 2012

# Psis = np.linspace(1,6,6)
# chis = [3e-3, 1.6e-3, 1e-3, 3.5e-4, 3e-4, 9e-5]#, 8e-6, 5e-7]
# lmax = 10
# mu = 0
# epsilon = 0.1

# fig2,ax2 = plt.subplots(1,1)

# for i in range(6):
#     print(Psis[i])
#     psi, r_grid, theta_grid = calculate_psi(lmax,epsilon,mu,Psis[i],chis[i], 1e-12, 1e-3)
    
#     density = rho_hat(psi)

#     ax2.plot(r_grid,density[0,:]/rho_hat(Psis[i]), label = '$\\theta =$ ' + str(theta_grid[0]), color = 'k', linestyle = '--', linewidth = 0.5)
#     ax2.plot(r_grid,density[6,:]/rho_hat(Psis[i]), label = '$\\theta =$ ' + str(theta_grid[6]), color = 'k', linewidth = 0.5)


# ax2.set_yscale('log')
# ax2.set_xscale('log')
# ax2.set_xlim(0.1,300)
# ax2.set_ylim(1e-8,1)
# #ax2.legend()
# ax2.set_ylabel('$\\rho/\\rho_0$')
# ax2.set_xlabel('$\\hat{r}$')

###### Development    
lmax = 10
epsilon = 0.1 
mu = 0
Psi = 7
chi = 3e-6

converged = False
tolerance = 1e-3
num_iters = 3
threshold = 1e-11

###Start of limited iteration number
Al = np.zeros((lmax+1,num_iters))
Bl = np.zeros((lmax+1,num_iters))
psi = []
u = []

u_n1, r_grid, theta_grid, psi_n1 = initialise_iteration([lmax, epsilon, mu, Psi, chi])

psi.append(psi_n1)
u.append(u_n1)

iteration = 0

while(iteration < num_iters):
    
    u_n, Al[:,iteration], Bl[:,iteration] = single_iter(u_n1, [lmax, epsilon, mu, Psi, chi], r_grid, theta_grid, threshold)
    psi_n = u_to_psi(u_n, r_grid, theta_grid, mu, chi)
    
    converged = stopping_condition(psi_n, psi_n1, tolerance)
    
    psi.append(psi_n)
    u.append(u_n)
    
    u_n1 = u_n
    psi_n1 = psi_n
        
    print(iteration)
    iteration += 1
    
# #### Problematic iteraiton identified as the 2nd, here is a careful step through of the second iteration.

psi_theta_r = u_to_psi(u[2], r_grid, theta_grid, mu, chi)
rho_theta_r = rho_hat(psi_theta_r)

rho_hat_1n = legendre_decomp(rho_theta_r,threshold)
 
r_1_minus_l_matrix = generate_r_power_matrix(1, 1-lmax, r_grid, lmax)
r_l_plus_2_matrix = generate_r_power_matrix(2, 2+lmax, r_grid, lmax)
r_matrix = np.tile(r_grid,(lmax + 1,1))

r_l_matrix = (1/r_1_minus_l_matrix)*r_matrix
r_l_plus_one = (1/r_l_plus_2_matrix) * r_matrix

integrand1 = r_1_minus_l_matrix * rho_hat_1n
integral1 = np.zeros(np.shape(integrand1))    
for i in range(np.shape(integrand1)[1]):
    integral1[:,i] =  simps(y = integrand1[:,0:i+1], x = r_grid[0:i+1], axis = 1)

integrand2 = r_l_plus_2_matrix * rho_hat_1n 
integral2 = np.zeros(np.shape(integrand2))    
for i in range(np.shape(integrand2)[1]):
    integral2[:,i] =  simps(y = integrand2[:,0:i+1], x = r_grid[0:i+1], axis = 1)
    
coeff_matrix = np.zeros((lmax+1, len(r_grid)))

for l in range(lmax+1):
    coeff_matrix[l,:] = 1/(2*l +1)

coeff_matrix *= -9/rho_hat(Psi) 

Al_via_infinity_behaviour = (-coeff_matrix * integral1)[:,-1]

particular_solution = coeff_matrix * (r_l_matrix * integral1 - r_l_plus_one * integral2) 

F_theta_r = np.zeros((len(theta_grid),len(r_grid)))

for i in range(len(r_grid)):    
    F_theta_r[:,i] = legval(x = np.cos(theta_grid), c = particular_solution[:,i])
    
A1 = np.zeros((lmax + 1,2*lmax + 2))
A2 = np.zeros((lmax + 1,2*lmax + 2))

I = np.identity(lmax+1)

for l in range(lmax+1):
    
    A1[:,l] = l * legval(x = np.cos(theta_grid), c = I[:,l])
    A1[:,lmax + 1 + l] = -(l+1) * legval(x = np.cos(theta_grid), c = I[:,l])
    
    A2[:,l] = legval(x = np.cos(theta_grid), c = I[:,l])
    A2[:,lmax + 1 + l] = legval(x = np.cos(theta_grid), c = I[:,l])
    
A = np.vstack((A1,A2))    
b  = np.zeros((2*lmax + 2,1))
b[lmax+1:, 0] = Psi - (9 * mu)/(4 * np.pi * epsilon)
#b[lmax+1:, 0] = Psi - (9 * mu)/(4 * np.pi * epsilon) - (9/2) * chi * epsilon**2 * np.sin(theta_grid)**2

coefficients = linalg.solve(A,b)
print(np.linalg.cond(A))
A_l_tild = coefficients[:lmax+1]
B_l_tild = coefficients[lmax+1:]

r_l = r_l_matrix
r_l1 = r_l_plus_one

A_l = A_l_tild / np.reshape(r_l[:,0],(lmax +1 ,1))
B_l = B_l_tild / np.reshape(r_l1[:,0],(lmax +1 ,1))

Al_matrix = np.tile(A_l, (1,len(r_grid)))
Bl_matrix = np.tile(B_l, (1,len(r_grid)))

final_coefficients = Al_matrix * r_l + Bl_matrix * r_l1 

homogeneous_solution = np.zeros((len(theta_grid),len(r_grid)))

for i in range(len(r_grid)):    
    homogeneous_solution[:,i] = legval(x = np.cos(theta_grid), c = final_coefficients[:,i])
    
u_single_iter_solution = homogeneous_solution + F_theta_r

psi_single_iter = u_to_psi(u_single_iter_solution, r_grid, theta_grid, mu, chi)

fig1,ax1 = plt.subplots(1,1)
ax1.plot(r_grid,psi_single_iter[0,:], label = '$\\theta =$ ' + str(theta_grid[0]))
ax1.plot(r_grid,psi_single_iter[-1,:], label = '$\\theta =$ ' + str(theta_grid[-1]))
ax1.plot(r_grid,psi_single_iter[6,:], label = '$\\theta =$ ' + str(theta_grid[6]))
# ax1.set_yscale('log')
# ax1.set_xscale('log')
ax1.legend()
ax1.set_ylabel('$\\psi$')
ax1.set_xlabel('$\\hat{r}$')    
    
###Testing legendre decomposition error
    
# num_trials = 100000
# max_errors = np.zeros(num_trials)

# for i in range(num_trials):
#     true_coeffs = np.random.uniform(low= 0, high = 100, size = lmax+1)
    
#     generated_polynomial = legval(x=np.cos(theta_grid), c = true_coeffs)
    
#     calculated_coeffs = flt.dlt(generated_polynomial, closed = True)
    
#     error = true_coeffs - calculated_coeffs
    
#     max_errors[i] = np.max(abs(error))
 
### Start of full iteration scheme
# u_n1, r_grid, theta_grid, psi_n1 = initialise_iteration([lmax, epsilon, mu, Psi, chi])

# u_n, Al, Bl = single_iter(u_n1, [lmax, epsilon, mu, Psi, chi], r_grid, theta_grid,threshold)
# psi_n = u_to_psi(u_n, r_grid, theta_grid, mu, chi)
# converged = stopping_condition(psi_n, psi_n1, tolerance)

# u_n1 = u_n
# psi_n1 = psi_n

# iteration = 1    

# while(converged == False):
    
#     u_n, Al, Bl = single_iter(u_n1, [lmax, epsilon, mu, Psi, chi], r_grid, theta_grid,threshold)
#     psi_n = u_to_psi(u_n, r_grid, theta_grid, mu, chi)
    
#     converged = stopping_condition(psi_n, psi_n1, tolerance)
    
#     u_n1 = u_n
#     psi_n1 = psi_n
    
#     print(iteration)
#     iteration += 1
    
# base_model = LoKi(mu, epsilon, Psi, pot_only = True)
# r_grid = base_model.rhat
# psi = base_model.psi
# ax1.plot(r_grid,psi, label = 'Loaded King model')

# fig1,ax1 = plt.subplots(1,1)
# ax1.plot(r_grid,psi_n[0,:], label = '$\\theta =$ ' + str(theta_grid[0]))
# ax1.plot(r_grid,psi_n[-1,:], label = '$\\theta =$ ' + str(theta_grid[-1]))
# ax1.plot(r_grid,psi_n[6,:], label = '$\\theta =$ ' + str(theta_grid[6]))
# ax1.set_yscale('log')
# ax1.set_xscale('log')
# ax1.set_xlim(epsilon,11)
# ax1.legend()
# ax1.set_ylabel('$\\psi$')
# ax1.set_xlabel('$\\hat{r}$')

# density = rho_hat(psi_n)

# fig2,ax2 = plt.subplots(1,1)
# ax2.plot(r_grid,density[0,:]/rho_hat(Psi), label = '$\\theta =$ ' + str(theta_grid[0]))
# ax2.plot(r_grid,density[6,:]/rho_hat(Psi), label = '$\\theta =$ ' + str(theta_grid[6]))
# ax2.set_yscale('log')
# ax2.set_xscale('log')
# ax2.set_xlim(0.1,300)
# ax2.set_ylim(1e-8,1)
# ax2.legend()
# ax2.set_ylabel('$\\rho/\\rho_0$')
# ax2.set_xlabel('$\\hat{r}$')


############### Developing single iteration

# lmax = 10
# epsilon = 0.01
# mu = 0
# Psi = 5
# chi = 1e-5
# constants = (lmax, epsilon, mu, Psi, chi)

# u_theta_r,r_grid,theta_grid, psi_theta_r = initialise_iteration(constants)
# rho_theta_r = rho_hat(psi_theta_r)

# rho_hat_1n = legendre_decomp(rho_theta_r,1e-10)

# r_1_minus_l_matrix = generate_r_power_matrix(1, 1-lmax, r_grid, lmax)
# r_l_plus_2_matrix = generate_r_power_matrix(2, 2+lmax, r_grid, lmax)
# r_matrix = np.tile(r_grid,(lmax + 1,1))

# r_l_matrix = (1/r_1_minus_l_matrix)*r_matrix
# r_l_plus_one = (1/r_l_plus_2_matrix) * r_matrix

# integrand1 = r_1_minus_l_matrix * rho_hat_1n
# integral1 = np.zeros(np.shape(integrand1))    
# for i in range(np.shape(integrand1)[1]):
#     integral1[:,i] =  simps(y = integrand1[:,0:i+1], x = r_grid[0:i+1], axis = 1)

# integrand2 = r_l_plus_2_matrix * rho_hat_1n 
# integral2 = np.zeros(np.shape(integrand2))    
# for i in range(np.shape(integrand2)[1]):
#     integral2[:,i] =  simps(y = integrand2[:,0:i+1], x = r_grid[0:i+1], axis = 1)
    
# coeff_matrix = np.zeros((lmax+1, len(r_grid)))

# for l in range(lmax+1):
#     coeff_matrix[l,:] = 1/(2*l +1)

# coeff_matrix *= -9/rho_hat(Psi) 

# Al_via_infinity_behaviour = (-coeff_matrix * integral1)[:,-1]

# particular_solution = coeff_matrix * (r_l_matrix * integral1 - r_l_plus_one * integral2) 

# F_theta_r = np.zeros((len(theta_grid),len(r_grid)))

# for i in range(len(r_grid)):    
#     F_theta_r[:,i] = legval(x = np.cos(theta_grid), c = particular_solution[:,i])
    
# A1 = np.zeros((lmax + 1,2*lmax + 2))
# A2 = np.zeros((lmax + 1,2*lmax + 2))

# I = np.identity(lmax+1)

# for l in range(lmax+1):
    
#     A1[:,l] = l * legval(x = np.cos(theta_grid), c = I[:,l])
#     A1[:,lmax + 1 + l] = -(l+1) * legval(x = np.cos(theta_grid), c = I[:,l])
    
#     A2[:,l] = legval(x = np.cos(theta_grid), c = I[:,l])
#     A2[:,lmax + 1 + l] = legval(x = np.cos(theta_grid), c = I[:,l])
    
# A = np.vstack((A1,A2))    
# b  = np.zeros((2*lmax + 2,1))
# b[lmax+1:, 0] = Psi - (9 * mu)/(4 * np.pi * epsilon)
# #b[lmax+1:, 0] = Psi - (9 * mu)/(4 * np.pi * epsilon) - (9/2) * chi * epsilon**2 * np.sin(theta_grid)**2

# coefficients = linalg.solve(A,b)
# print(np.linalg.cond(A))
# A_l_tild = coefficients[:lmax+1]
# B_l_tild = coefficients[lmax+1:]

# r_l = r_l_matrix
# r_l1 = r_l_plus_one

# A_l = A_l_tild / np.reshape(r_l[:,0],(lmax +1 ,1))
# B_l = B_l_tild / np.reshape(r_l1[:,0],(lmax +1 ,1))

# Al_matrix = np.tile(A_l, (1,len(r_grid)))
# Bl_matrix = np.tile(B_l, (1,len(r_grid)))

# final_coefficients = Al_matrix * r_l + Bl_matrix * r_l1 

# homogeneous_solution = np.zeros((len(theta_grid),len(r_grid)))

# for i in range(len(r_grid)):    
#     homogeneous_solution[:,i] = legval(x = np.cos(theta_grid), c = final_coefficients[:,i])
    
# u_single_iter_solution = homogeneous_solution + F_theta_r

# psi_single_iter = u_to_psi(u_single_iter_solution, r_grid, theta_grid, mu, chi)

# fig1,ax1 = plt.subplots(1,1)
# ax1.plot(r_grid,psi_single_iter[0,:], label = '$\\theta =$ ' + str(theta_grid[0]))
# ax1.plot(r_grid,psi_single_iter[-1,:], label = '$\\theta =$ ' + str(theta_grid[-1]))
# ax1.plot(r_grid,psi_single_iter[1,:], label = '$\\theta =$ ' + str(theta_grid[1]))
# ax1.set_yscale('log')
# ax1.set_xscale('log')
# ax1.legend()
# ax1.set_ylabel('$\\psi$')
# ax1.set_xlabel('$\\hat{r}$')

######## Testing quadrature solution

# r_grid = np.linspace(1,10,1000)
# lmax = 5
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


# l_vector = np.array([0,1,2,3,4,5])
# one = np.ones((n_samp,len(r_grid)))
# l_matrix1 = np.zeros((n_samp,len(r_grid)))
# integration_1 = np.zeros((n_samp,len(r_grid)))

# for i in range(len(r_grid)):
#     l_matrix1[:,i] = 1/(3-l_vector)
#     integration_1[:,i] = r_grid[i]**(3-l_vector)
# result_of_integration1 = l_matrix1 * (integration_1 - one)

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
    




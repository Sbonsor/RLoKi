#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:10:49 2023

@author: s1984454
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import gammainc
from scipy.special import gamma
from scipy.integrate import solve_ivp

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

def rho_hat(psi):
    
   if(psi > 0):
       
       density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
       
   else:
       
       density = 0
 
   return density

def ODEs(r,y,Psi,mu,epsilon):
    
    RHS= np.zeros(2,)
    RHS[0] = y[1]
    RHS[1] = -(2/r)*y[1] -9*(rho_hat(y[0])/rho_hat(Psi)) - (27*mu/(4*np.pi*epsilon**3)) * (1 + (r/epsilon)**2) ** (-5/2)
    
    return RHS

def zero_cross(r,y, Psi, mu, epsilon):
    return y[0] 
zero_cross.terminal = True

def solve_ODEs(Psi,mu,epsilon, final_radius = 1e9 ):
    
    solution = solve_ivp(fun = ODEs, t_span = (1e-6,final_radius), y0 = (Psi,0),method = 'RK45', dense_output = True,rtol = 1e-8, atol = 1e-30,events = zero_cross, args = [Psi,mu,epsilon])
    
    psi = solution.y[0,:]
    psi_grad = solution.y[1,:]
    rhat = solution.t
    
    return psi, psi_grad, rhat

# Psi = 5
# mu = 0.69
# epsilon = 0.1

# psi, psi_grad, rhat = solve_ODEs(Psi,mu,epsilon, final_radius = 1e9 )

# fig1,ax1 = plt.subplots(1,1)
# ax1.plot(rhat,psi)
# ax1.set_xlabel('$\hat{r}$')
# ax1.set_ylabel('$\psi$')

############## rt vs mu for fixed Psi,epsilon

Psi = 5
epsilon = 0.1
n_mus = 700

max_mu = 4*np.pi*Psi*epsilon/9
mus = np.linspace(0,max_mu,n_mus)

rts = []

i = 0
for mu in mus:
    
    psi, psi_grad, rhat = solve_ODEs(Psi,mu,epsilon, final_radius = 1e9 )
    rts.append(rhat[-1])
    i += 1
    print(i)
    
fig2, ax2 = plt.subplots(1,1)
ax2.plot(mus/max_mu, rts)
ax2.set_xlabel('$\\mu/\\mu_{max}$')
ax2.set_ylabel('$\\hat{r}_t$')
ax2.set_title(f'$(\\Psi, \\epsilon) = ({Psi}, {epsilon})$')    
    
    
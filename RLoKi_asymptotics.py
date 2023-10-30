#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:54:19 2022

@author: s1984454
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
from LoKi import LoKi
from RLoKi import RLoKi
from numpy.polynomial.legendre import legval

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

class RLoKi_asymp:
    
    def __init__(self, mu, epsilon, Psi, chi, **kwargs ):
        
        self._set_kwargs(mu, epsilon, Psi, chi, **kwargs)
        self.solve_odes()
            
    def _set_kwargs(self, mu, epsilon, Psi, chi, **kwargs):
        
        if (Psi - 9*mu/(4*np.pi*epsilon) < 0):
            raise ValueError("a_0 must not be negative")
            
        ### Dimensionless parameters
        self.Psi = Psi
        self.mu = mu
        self.epsilon = epsilon
        self.chi = chi
                
        ### Parameters required for numerics
        self.ode_rtol = 1e-8
        self.ode_atol = 1e-30
        self.r_final = 1e8

        
        if kwargs is not None:
            for key,value in kwargs.items():
                setattr(self,key,value)
                
    def inc_gamma(self, a, psi):
        
        return gammainc(a, psi)*gamma(a)

    def rho_hat(self,psi):
        
        if (psi>0):
            
            return np.exp(psi)*self.inc_gamma(5/2,psi)
        
        else:
            
            return 0

    def R1(self, psi):
        
        if (psi > 0):
            
            R1 = (9/self.rho_hat(self.Psi)) * (np.exp(psi)*self.inc_gamma(5/2,psi) + np.power(psi,3/2))
            
        else:
            return 0
        
        return R1

    def odes(self,r, y):
            
        ode = np.zeros(6)
        
        ode[0] = y[1]
        ode[1] = -2*y[1]/r -9*self.rho_hat(y[0] + (9*self.mu)/(4*np.pi*r)) / self.rho_hat(self.Psi)
        
        ode[2] = y[3]
        ode[3] = -2*y[3]/r - self.R1(y[0] + (9*self.mu)/(4*np.pi*r)) * y[2] + 18
        
        ode[4] = y[5]
        ode[5] = -2*y[5]/r - self.R1(y[0] + (9*self.mu)/(4*np.pi*r)) * y[4] + (6/r**2) * y[4] 
        
        # u_0 = y[0]
        # u_10 = y[2]
        # gamma_2 = y[4]
        
        return ode

    def ics(self):
        
        ic = np.zeros(6)
        
        ic[0] = self.Psi - (9*self.mu)/(4*np.pi*self.epsilon)
        ic[1] = 0
        
        ic[2] = 3 * self.epsilon**2
        ic[3] = 6 * self.epsilon
        
        ic[4] = self.epsilon**2
        ic[5] = 2 * self.epsilon
        
        return ic

    def psi_0_crossing(self, r, y):
        return y[0] + (9 * self.mu) / (4 * np.pi * r)
    psi_0_crossing.terminal = True

    def psi_1_crossing(self, r, y):
        
        psi0_grad = y[1] - (9 * self.mu) / (4 * np.pi * r**2)
        psi1_grad = y[3] + self.A_2 * y[5] * 0.5 * (3 * np.cos(np.pi/2)**2 - 1)
        
        return psi0_grad + self.chi * psi1_grad - 1e-8
    psi_1_crossing.terminal = True
    
                
    def solve_odes(self):
        
        initial_conditions = self.ics()
        
        sol_to_rt = solve_ivp(fun = self.odes, t_span = (self.epsilon, self.r_final), y0 = initial_conditions, method = 'RK45', rtol = self.ode_rtol, atol = self.ode_atol, events = self.psi_0_crossing)
        
        self.rt = sol_to_rt.t[-1]
        self.psi_0_grad_rt = sol_to_rt.y[1,-1] - (9*self.mu)/(4*np.pi*self.rt**2) 
        self.u_10_rt = sol_to_rt.y[2,-1]
        self.u_10_grad_rt = sol_to_rt.y[3,-1]
        self.gamma_2_rt = sol_to_rt.y[4,-1]
        self.gamma_2_grad_rt = sol_to_rt.y[5,-1]
        
        self.alpha_0 = self.rt * self.psi_0_grad_rt
        self.lambda_0 = self.rt**2 * self.psi_0_grad_rt
        self.lambda_1 = self.u_10_grad_rt * self.rt**2 - 6*self.rt**3
        self.alpha_1 =  self.u_10_rt + self.lambda_1/self.rt - 3*self.rt**2
        self.A_2 = (-15 * self.rt**2) / (self.rt * self.gamma_2_grad_rt + 3 * self.gamma_2_rt)
        self.a_2 = -self.rt**3 * (self.A_2 * self.gamma_2_rt + 3*self.rt**2)

        sol_from_rt = solve_ivp(fun = self.odes, t_span = (self.rt, self.rt*3), y0 = sol_to_rt.y[:,-1] , method = 'RK45', rtol = self.ode_rtol, atol = self.ode_atol, events = self.psi_1_crossing)

        sol_to_rt.t = np.reshape(sol_to_rt.t, (len(sol_to_rt.t),1))
        sol_from_rt.t = np.reshape(sol_from_rt.t, (len(sol_from_rt.t),1))

        rhat = np.concatenate( (sol_to_rt.t, sol_from_rt.t), axis = 0)
        
        full_solution = np.concatenate( (sol_to_rt.y, sol_from_rt.y), axis = 1)
        
        self.u_0 = full_solution[0,:]
        self.u_0_grad = full_solution[1,:]
        self.u_10 = full_solution[2,:]
        self.u_10_grad = full_solution[3,:]
        self.gamma_2 = full_solution[4,:]
        self.gamma_2_grad = full_solution[5,:]
        self.rhat = rhat.ravel()
        
        # rhat = sol.t
        # u_0 = sol.y[0,:]
        # u_10 = sol.y[2,:]
        # gamma_2 = sol.y[4,:]

        return 1
    
    def psi_theta_r(self, theta):

        psi0 = self.u_0 + (9 * self.mu) / (4 * np.pi * self.rhat)
        psi1 = self.u_10 + self.A_2 * self.gamma_2 * 0.5 * (3 * np.cos(theta)**2 - 1)
        
        return psi0 + self.chi * psi1
    
    def psi_grad_theta_r(self, theta):

        psi0_grad = self.u_0_grad - (9 * self.mu) / (4 * np.pi * self.rhat**2)
        psi1_grad = self.u_10_grad + self.A_2 * self.gamma_2_grad * 0.5 * (3 * np.cos(theta)**2 - 1)
        
        return psi0_grad + self.chi * psi1_grad

def determine_critical_chi_asymp(mu,epsilon, Psi, step_size_criterion =  3.90625e-05):
    
    chis = []
    loki_model = LoKi(mu, epsilon, Psi)
    
    lambda_0 = loki_model.dpsi_dr[-1] * loki_model.rt**2
    alpha_0 = lambda_0/loki_model.rt
    chi_crit_estimate = (-8*alpha_0**3) / (243*lambda_0**2)
    
    upper_chi = 10*chi_crit_estimate
    lower_chi = 0.1* chi_crit_estimate
    
    step_size = 0.5*(np.log10(upper_chi) - np.log10(lower_chi))
    
    while(step_size > step_size_criterion):
        
            new_chi = 10**(np.log10(upper_chi) - step_size)
            chis.append(new_chi)
            model = RLoKi_asymp(mu, epsilon, Psi, new_chi)
                        
            psi_equatorial = model.psi_theta_r(np.pi/2)
            #plt.plot(model.rhat, model.psi_theta_r(np.pi/2))
            gradient_estimate = gradient_estimate = model.psi_grad_theta_r(np.pi/2)
            
            grad_idx = np.where(gradient_estimate < 0)[0][-1]
            indices_for_interpolation = range(grad_idx-20, min(grad_idx+2, len(gradient_estimate)))
            
            r_grad0 = np.interp(0, gradient_estimate[indices_for_interpolation], model.rhat[indices_for_interpolation] )
            
            psi_at_turning_point = np.interp(r_grad0, model.rhat[indices_for_interpolation], psi_equatorial[indices_for_interpolation])
            
            if(psi_at_turning_point > 0):
                
                upper_chi = new_chi
                
            if(psi_at_turning_point < 0):
                
                lower_chi = new_chi
                last_chi = new_chi
                last_model = model
                rb = r_grad0
            
            step_size = 0.5*(np.log10(upper_chi) - np.log10(lower_chi))
                
    return chis, last_chi, last_model,rb


# Psi = 5.14288
# mu = 0
# epsilon = 0.01
# chi = 0.00195166


# ode_rtol = 1e-8
# ode_atol = 1e-30

# model = RLoKi_asymp(mu, epsilon, Psi, chi, ode_rtol = ode_rtol, ode_atol = ode_atol)

# psi_equatorial = model.psi_theta_r(np.pi/2)
# psi_polar = model.psi_theta_r(0)

# fig1, ax1 = plt.subplots(1,1)
# ax1.plot(model.rhat,psi_equatorial, label = 'Asymptotic solution equatorial', color = 'k')
# #ax1.plot(model.rhat,psi_polar, label = 'Asymptotic solution polar', color = 'r')
# ax1.axhline(y = 0, linestyle = '--')
# ax1.set_xlim(epsilon, model.rhat[-1])
# ax1.set_ylim(-0.01,Psi)
# #ax1.set_xscale('log')
# #ax1.set_yscale('log')


# psi_1 = model.u_10 + model.A_2 * model.gamma_2 * 0.5 * (3 * np.cos(np.pi/2)**2 - 1)
# A_2 = model.A_2
# gamma_2 = model.gamma_2
# u_10 = model.u_10
# u_0 = model.u_0
# psi_0 = model.u_0 + (9 * model.mu) / (4 * np.pi * model.rhat)

# # RLoki_model = RLoKi(mu, epsilon, Psi, chi)
# # rloki_equatorial = RLoki_model.interp_equatorial_plane()

# # ax1.plot(RLoki_model.r_grid, rloki_equatorial, label = 'Iterative solution equatorial', color = 'k', linestyle = '--')
# # ax1.plot(RLoki_model.r_grid, RLoki_model.psi[0,:], label = 'Iterative solution polar', color = 'r', linestyle = '--')

# # ax1.legend()

# loki_model = LoKi(mu, epsilon, Psi)

# condition = psi_equatorial <= 0

# if(len(psi_equatorial[condition]) == 0):
#     flag = 0
#     print('Super-critical')
# else:
#     flag = 1
#     print('Sub-critical')
    






















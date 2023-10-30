#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:16:07 2023

@author: s1984454
"""
import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt
from RLoKi_asymptotics import RLoKi_asymp
from LoKi import LoKi
import pickle

def determine_critical_chi_asymp(mu,epsilon, Psi, step_size_criterion):
    
    chis = []
    loki_model = LoKi(mu, epsilon, Psi)
    
    lambda_0 = loki_model.dpsi_dr[-1] * loki_model.rt**2
    alpha_0 = lambda_0/loki_model.rt
    chi_crit_estimate = (-8*alpha_0**3) / (243*lambda_0**2)
    
    upper_chi = chi_crit_estimate
    lower_chi = 0.1* chi_crit_estimate
    
    step_size = 0.5*(np.log10(upper_chi) - np.log10(lower_chi))
    
    while(step_size > step_size_criterion):
        
            new_chi = 10**(np.log10(upper_chi) - step_size)
            chis.append(new_chi)
            model = RLoKi_asymp(mu, epsilon, Psi, new_chi)
                        
            psi_equatorial = model.psi_theta_r(np.pi/2)
            plt.plot(model.rhat, model.psi_theta_r(np.pi/2))
            gradient_estimate = np.gradient(psi_equatorial, model.rhat)
            
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
            
            step_size = 0.5*(np.log10(upper_chi) - np.log10(lower_chi))
                
    return chis, last_chi, last_model

Psi = 5
mu = 0
epsilon = 1e-5
step_size_criterion = 3.90625e-05/16

chis = []
loki_model = LoKi(mu, epsilon, Psi)

lambda_0 = loki_model.dpsi_dr[-1] * loki_model.rt**2
alpha_0 = lambda_0/loki_model.rt
chi_crit_estimate = (-8*alpha_0**3) / (243*lambda_0**2)

upper_chi = 2*chi_crit_estimate
lower_chi = 0.1* chi_crit_estimate

step_size = 0.5*(np.log10(upper_chi) - np.log10(lower_chi))

while(step_size > step_size_criterion):
    
        new_chi = 10**(np.log10(upper_chi) - step_size)
        chis.append(new_chi)
        model = RLoKi_asymp(mu, epsilon, Psi, new_chi)
                    
        psi_equatorial = model.psi_theta_r(np.pi/2)
        plt.plot(model.rhat, model.psi_theta_r(np.pi/2))
        gradient_estimate = model.psi_grad_theta_r(np.pi/2)
        
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
        
        step_size = 0.5*(np.log10(upper_chi) - np.log10(lower_chi))

fig,ax = plt.subplots(1,1)
ax.plot(model.rhat,psi_equatorial)
ax.plot(model.rhat, gradient_estimate)
ax.plot(model.rhat[indices_for_interpolation], gradient_estimate[indices_for_interpolation])
ax.axhline(y=0, color = 'k', linestyle = '--')
ax.axhline(y = psi_at_turning_point, color = 'r', linestyle = '--')
ax.axvline(x = r_grad0, color = 'k', linestyle = '--')
ax.set_ylim(-0.5,0.5)










#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:15:43 2022

@author: s1984454
"""
from LoKi import LoKi
import numpy as np
from numpy.polynomial.legendre import legval
import flt
from scipy.special import gammainc,gamma
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator

class RLoKi:
    
    def __init__(self, mu, epsilon, Psi, chi, **kwargs ):
        
        self._set_kwargs(mu, epsilon, Psi, chi, **kwargs)
        
        self.psis, self.us, self.Als, self.rho_hat_coeffs = self.run_iteration_scheme()
        
        self.psi = self.psis[-1]
        self.u = self.us[-1]
    
    
    def _set_kwargs(self, mu, epsilon, Psi, chi, **kwargs):
        
        if (Psi - 9*mu/(4*np.pi*epsilon) < 0):
            raise ValueError("a_0 must not be negative")
            
        ### Dimensionless parameters
        self.Psi = Psi
        self.mu = mu
        self.epsilon = epsilon
        self.chi = chi
                
        ### Parameters required for numerics
        self.lmax = 6
        self.legendre_threshold = 1e-11## Need to figure out a principled way to set this
        self.max_iters = 60
        self.tol = 1e-3
        
        if kwargs is not None:
            for key,value in kwargs.items():
                setattr(self,key,value)
                
    def legendre_decomp(self,u):
        
        coeffs = np.zeros(np.shape(u))
        
        for i in range(np.shape(u)[1]):
            coeffs[:,i] = flt.dlt(u[:,i], closed = True)
        
        coeffs[abs(coeffs) < self.legendre_threshold] = 0
        
        return coeffs

    def initialise(self):
        
        self.theta_grid = flt.theta(self.lmax+1, closed = True)
        
        base_model = LoKi(self.mu, self.epsilon, self.Psi, pot_only = True)
        
        rt = base_model.rhat[-1]
        
        inner_grid = base_model.rhat
        spacing = base_model.rhat[-1] - base_model.rhat[-2]
        outer_grid = np.arange(rt + spacing, 2*rt, spacing)
        
        self.r_grid = np.concatenate((inner_grid,outer_grid))
        
        psi = np.interp(self.r_grid, base_model.rhat, base_model.psi)
        psi_theta_r = np.tile(psi,(self.lmax+1,1))
        
        
        u_theta_r = self.psi_to_u(psi_theta_r)

        return u_theta_r, psi_theta_r

    def rho_hat(self, psi):
        
        density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
        density = np.nan_to_num(density,copy = False)
            
        return density

    def u_to_psi(self, u):
        
        R,Theta = np.meshgrid(self.r_grid, self.theta_grid)
        
        psi = u + 9*self.mu/(4*np.pi*R) + (9/2)* self.chi * R**2 * np.sin(Theta)**2
        
        return psi

    def psi_to_u(self, psi):
        
        R,Theta = np.meshgrid(self.r_grid, self.theta_grid)
        
        u = psi - 9*self.mu/(4*np.pi*R) - (9/2)* self.chi * R**2 * np.sin(Theta)**2
        
        return u

    def r_power_matrix(self, power_start, power_end):
        
        r_power_matrix = np.zeros((self.lmax+1, len(self.r_grid)))
        powers = np.linspace(power_start,power_end, self.lmax+1)
        
        for i in range(self.lmax+1):
            r_power_matrix[i,:] = np.power(self.r_grid, powers[i])
            
        return  r_power_matrix

    def integrate_matrix(self,integrand):
        
        result = np.zeros(np.shape(integrand))
        
        for i in range(np.shape(integrand)[1]):
            result[:,i] =  simps(y = integrand[:,0:i+1], x = self.r_grid[0:i+1])
        
        return result

    def coefficient_matrix(self):
        
        coeffs = np.zeros((self.lmax+1, len(self.r_grid)))
        
        for l in range(self.lmax+1):
            
            coeffs[l,:] = 1/(2*l +1)
        
        coeffs *= -9  / self.rho_hat(self.Psi)
        
        return coeffs

    def sum_f_l_r(self, f_l_r):
        
        result = np.zeros((len(self.theta_grid),len(self.r_grid)))

        for i in range(len(self.r_grid)):    
            result[:,i] = legval(x = np.cos(self.theta_grid), c = f_l_r[:,i])
        
        return result

    def stopping_condition(self, psi_n, psi_n1):
        
        #criterion = np.nan_to_num(np.max(abs((psi_n - psi_n1)/psi_n)))
        idx = psi_n != 0
        criterion = np.max(abs((psi_n[idx] - psi_n1[idx])/psi_n[idx]))

        if criterion < self.tol:
            
            return True
        
        else:
            return False

    def single_iteration(self, psi_theta_r_n1, u_theta_r_n1):
        
        rho_theta_r_n1 = self.rho_hat(psi_theta_r_n1)

        rho_hat_n1 = self.legendre_decomp(rho_theta_r_n1)

        coeff_matrix = self.coefficient_matrix()

        ######Should really pre-calculate these rather than redoing them at every step.
        r1l_matrix = self.r_power_matrix(1, 1 - self.lmax)
        rl2_matrix = self.r_power_matrix(2, 2 + self.lmax)
        r_matrix = np.tile(self.r_grid,(self.lmax + 1,1))

        rl_matrix = r_matrix / r1l_matrix
        r_l1_matrix = r_matrix / rl2_matrix

        integrand1 = r1l_matrix * rho_hat_n1
        integrand2 = rl2_matrix * rho_hat_n1

        integral1 = coeff_matrix * self.integrate_matrix(integrand1)
        integral2 = coeff_matrix * self.integrate_matrix(integrand2)

        Al = -integral1[:,-1]
        Al[0] = self.Psi - (9*self.mu)/(4*np.pi*self.epsilon)
        Al_matrix = np.tile(np.reshape(Al,(len(Al),1)), (1,len(self.r_grid)))

        particular_coefficients = integral1 * rl_matrix - integral2 * r_l1_matrix
        particular_solution = self.sum_f_l_r(particular_coefficients)

        homogeneous_coefficients = Al_matrix * rl_matrix 
        homogeneous_solution = self.sum_f_l_r(homogeneous_coefficients)

        u_theta_r_n = homogeneous_solution + particular_solution
        psi_theta_r_n = self.u_to_psi(u_theta_r_n)
        
        return u_theta_r_n, psi_theta_r_n, Al, rho_hat_n1

    def run_iteration_scheme(self):
        
        print(self.chi)
        print(self.Psi)
        print(self.epsilon)
        print(self.mu)
        u_theta_r_n1, psi_theta_r_n1 = self.initialise()
        
        psis = []
        Als = []
        us = []
        rho_hat_coeffs = []
        
        psis.append(psi_theta_r_n1)
        us.append(u_theta_r_n1)
        
        self.converged = False
        iteration = 0
        
        while((self.converged == False) and (iteration < self.max_iters)):
            
            u_theta_r_n, psi_theta_r_n, Al,rho_hat_n1 = self.single_iteration(psi_theta_r_n1, u_theta_r_n1)
            
            psis.append(psi_theta_r_n)
            Als.append(Al)
            us.append(u_theta_r_n)
            rho_hat_coeffs.append(rho_hat_n1)
            
            self.converged = self.stopping_condition(psi_theta_r_n, psi_theta_r_n1)
            
            iteration += 1
            
            #print(iteration)
            
            if(self.converged == True):
                print('Solution converged!')
            
            if(iteration == self.max_iters):
                print('Solution not converged in specified number of iterations.')
                
            u_theta_r_n1 = u_theta_r_n
            psi_theta_r_n1 = psi_theta_r_n
            
        
        return psis, us, Als, rho_hat_coeffs
    
    def interp_to_theta(self,theta):
        psi_theta_r = self.psi
        
        interpolator = RegularGridInterpolator((self.theta_grid, self.r_grid), psi_theta_r)

        psi_interp =  interpolator((theta, self.r_grid))
        
        return psi_interp
    
def determine_critical_chi(mu,epsilon, Psi, m_iters, step_size, step_size_criterion):
    chis = []
    loki_model = LoKi(mu, epsilon, Psi)
    
    lambda_0 = loki_model.dpsi_dr[-1] * loki_model.rt**2
    alpha_0 = lambda_0/loki_model.rt
    chi_crit_estimate = (-8*alpha_0**3) / (243*lambda_0**2)
    
    chi_n = chi_crit_estimate # 10 ** np.log10(chi_crit_estimate*1.1)
    last_chi = chi_crit_estimate
    
    
    while(step_size > step_size_criterion):
        chis.append(chi_n)
        model = RLoKi(mu, epsilon, Psi, chi_n, max_iters = m_iters)
        
        if(model.converged == False):
            print('Not converged!')
            step_size = 0.5*step_size
            chi_n = 10 ** (np.log10(chi_n) - step_size)
    
        else:
            
            psi_equatorial = model.interp_to_theta(np.pi/2)
            
            gradient_estimate = np.gradient(psi_equatorial, model.r_grid)
            
            grad_idx = np.where(gradient_estimate[1:-1] < 0)[0][-1]
            indices_for_interpolation = range(grad_idx-20, min(grad_idx+2, len(gradient_estimate)))
            
            r_grad0 = np.interp(0, gradient_estimate[indices_for_interpolation], model.r_grid[indices_for_interpolation] )
            
            psi_at_turning_point = np.interp(r_grad0, model.r_grid[indices_for_interpolation], psi_equatorial[indices_for_interpolation])
            
            if(psi_at_turning_point > 0):
                step_size = step_size * 0.5
                chi_n = 10 ** (np.log10(chi_n) - step_size)
                
            if(psi_at_turning_point < 0):
                last_chi = chi_n
                last_model = model
                rb = r_grad0
                chi_n = 10 ** (np.log10(chi_n) + step_size)
                
    return chis, last_chi, last_model, rb
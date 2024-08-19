#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:49:42 2024

@author: s1984454
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from RLoKi_asymptotics import determine_critical_chi_asymp, RLoKi_asymp
from RLoKi import RLoKi
import pickle
from scipy.integrate import simpson
plt.rc('text', usetex=True)
plt.rcParams['font.size'] = 16

##### Fig. 3.1

# Psi = 5
# mu = 0.2
# epsilon = 0.1

# _, chi, _, _ = determine_critical_chi_asymp(mu,epsilon, Psi, step_size_criterion =  3.90625e-05)

# model = RLoKi_asymp(mu, epsilon, Psi, chi)

# psi_equatorial = model.psi_theta_r(np.pi/2)
# psi_polar = model.psi_theta_r(0)

# RLoki_model = RLoKi(mu, epsilon, Psi, chi)
# rloki_equatorial = RLoki_model.interp_to_theta(theta = np.pi/2)
# rloki_polar = RLoki_model.interp_to_theta(theta = 0)


# fig1, ax1 = plt.subplots(1,1)


# ax1.plot(RLoki_model.r_grid, rloki_equatorial, label = 'Iteration equatorial', color = 'r')
# ax1.plot(RLoki_model.r_grid, rloki_polar, label = 'Iteration  polar', color = 'r', linestyle = '--')

# ax1.plot(model.rhat,psi_equatorial, label = 'Asymptotic equatorial', color = 'k')
# ax1.plot(model.rhat,psi_polar, label = 'Asymptotic polar', color = 'k', linestyle = '--')

# ax1.legend(loc = 'upper right')
# ax1.set_title(f'$(\\Psi,\\mu,\\epsilon) = $({Psi}, {mu}, {epsilon})')


##### Fig. 3.2

# def psi_profiles_fixed_Psi_epsilon_mu(Psi, mu, epsilon, num_chis):
#     _, chi_crit, _, r_b = determine_critical_chi_asymp(mu,epsilon, Psi, step_size_criterion =  3.90625e-05/16)
    
#     chis = np.linspace(0,chi_crit,num_chis)
    
#     fig,ax = plt.subplots(1,1)
#     ax.set_ylabel('$\\psi$')
#     ax.set_xlabel('$\\hat{r}$')
#     axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    
#     linestyles = ['solid', 'dotted', 'dashdot', '-', '--']
#     i=0
#     for chi in chis:
        
#         model = RLoKi_asymp(mu, epsilon, Psi, chi)
#         ax.plot(model.rhat, model.psi_theta_r(np.pi/2), label = str(i/(num_chis-1)) +'$\\chi_{c}$', linestyle = linestyles[i])
#         axins.plot(model.rhat, model.psi_theta_r(np.pi/2), linestyle = linestyles[i])
#         i += 1
        
#     ax.set_xlim(epsilon,model.rhat[-1])    
#     ax.set_ylim(1e-8,Psi)
#     ax.set_xlim(0, 18)
#     ax.set_title(f'$(\\Psi, \\mu, \\epsilon) = ({Psi},{mu},{epsilon})$')
#     ax.legend(loc = 'lower left')
    
#     #ax.set_xscale('log')
#     #ax.set_yscale('log')
     
#     # subregion of the original image
#     x1, x2, y1, y2 = 8, 18, -0.5, 0.6
#     axins.set_xlim(x1, x2)
#     axins.set_ylim(y1, y2)
#     axins.set_xticklabels([])
#     axins.set_yticklabels([])
#     axins.axhline(y=0, linestyle = '--', color = 'k', linewidth = 0.2)
    
#     ax.indicate_inset_zoom(axins, edgecolor="black")
    
#     return 1

# Psi = 5
# mu = 0
# epsilon = 0.1
# num_chis = 5

# psi_profiles_fixed_Psi_epsilon_mu(Psi, mu, epsilon, num_chis)

##### Fig. 3.3

# def crit_chi_against_mu(Psi, epsilon, nsamp, step_size_criterion = 3.90625e-05, calculate = True):
    
#     if(calculate == False):
        
#         with open('Data/asymp_chi_crit_Psi_' + str(Psi) + '_eps_' + str(epsilon) + '.pkl', 'rb') as f:
#             results_dict = pickle.load(f)
    
#     else:    
        
#         mu_max = 4*np.pi*epsilon*Psi/9
#         mus = np.linspace(0, mu_max,nsamp)
        
#         chi_crits = np.zeros(nsamp)
        
#         for i in range(nsamp):
#             mu = mus[i]
#             chis,last_chi,last_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi, step_size_criterion)
#             chi_crits[i] = last_chi
            
#         results_dict = {'chi_crits' : chi_crits,'mus' : mus, 'Psi_epsilon_stepsize_criterion' : [Psi,epsilon, step_size_criterion]}
        
#         with open('Data/asymp_chi_crit_Psi_' + str(Psi) + '_eps_' + str(epsilon) + '.pkl', 'wb') as f:
#             pickle.dump(results_dict, f)
               
#     chi_crits = results_dict['chi_crits']
#     mus = results_dict['mus']
#     args = results_dict['Psi_epsilon_stepsize_criterion']
    
#     fig1,ax1 = plt.subplots()
#     ax1.plot(mus, chi_crits)
#     ax1.set_xlabel('$\\mu$')
#     ax1.set_ylabel('$\\chi_{c}$')
#     ax1.set_title('$(\\Psi, \\epsilon) =$ '+ '(' + str(args[0]) + ', '+ str(args[1])+')')
#     ax1.set_yscale('log')
      
#     return 1

# Psis = [3,5,7]
# epsilon = 0.1
# step_size_criterion = 3.90625e-05
# nsamp = 500
# calculate = True

# for Psi in Psis:
#     crit_chi_against_mu(Psi, epsilon, nsamp, step_size_criterion = step_size_criterion, calculate = calculate)
#     print(Psi)

# fig,ax = plt.subplots(1,1)
# ax.set_xlabel('$\\mu/\\mu_{max}$')
# ax.set_ylabel('$\\chi_{c}$')
# ax.set_yscale('log')
# ax.set_xlim(0,1)

# linestyles = ['solid', 'dotted', 'dashdot']
# i = 0
# for Psi in Psis:
    
#     with open('Data/asymp_chi_crit_Psi_' + str(Psi) + '_eps_' + str(epsilon) + '.pkl', 'rb') as f:
#         results_dict = pickle.load(f)
        
#     chi_crits = results_dict['chi_crits']
#     mus = results_dict['mus']
#     mu_max = 4*np.pi*epsilon*Psi/9
#     ax.plot(mus/mu_max,chi_crits, label = f'$\\Psi$ = {Psi}', linestyle = linestyles[i])
#     ax.legend(loc = 'lower left')
#     i = i+1

    
##### Fig. 3.4

# def crit_chi_against_a0(mu, epsilon, a0_min, a0_max, nsamp, step_size_criterion = 3.90625e-05, calculate = True):
    
#     if(calculate == False):
        
#         with open('Data/asymp_chi_crit_mu_' + str(mu) + '_eps_' + str(epsilon) + '_a0max_' + str(a0_max) + '.pkl', 'rb') as f:
#             results_dict = pickle.load(f)
    
#     else:    
        
#         a0s = np.linspace(a0_min, a0_max,nsamp)
#         Psis = a0s + (9*mu)/(4*np.pi*epsilon)
        
#         chi_crits = np.zeros(nsamp)
        
#         for i in range(nsamp):
#             Psi = Psis[i]
#             chis,last_chi,last_model, rb = determine_critical_chi_asymp(mu,epsilon, Psi, step_size_criterion)
#             chi_crits[i] = last_chi
#             print(i)
            
#         results_dict = {'chi_crits' : chi_crits,'a0s' : a0s,'Psis': Psis, 'mu_epsilon_stepsize_criterion' : [mu,epsilon, step_size_criterion]}
        
#         with open('Data/asymp_chi_crit_mu_' + str(mu) + '_eps_' + str(epsilon) + '_a0max_' + str(a0_max) + '.pkl', 'wb') as f:
#             pickle.dump(results_dict, f)
               
#     chi_crits = results_dict['chi_crits']
#     a0s = results_dict['a0s']
#     args = results_dict['mu_epsilon_stepsize_criterion']
    
#     fig1,ax1 = plt.subplots()
#     ax1.plot(a0s, chi_crits)
#     ax1.set_xlabel('$a_0$')
#     ax1.set_ylabel('$\\chi_{c}$')
#     ax1.set_title('$(\\mu, \\epsilon) =$ '+ '(' + str(args[0]) + ', '+ str(args[1])+')')
#     ax1.set_yscale('log')

    
#     return 1

# mus = [0,0.1,0.3]
# epsilon = 0.1
# a0_min = 0.01
# a0_max = 7
# step_size_criterion = 3.90625e-05
# nsamp = 500
# calculate = False

# for mu in mus:
#     crit_chi_against_a0(mu, epsilon, a0_min, a0_max, nsamp, step_size_criterion = step_size_criterion, calculate = calculate)
#     print(mu)

# fig,ax = plt.subplots(1,1)
# ax.set_xlabel('$a_0$')
# ax.set_ylabel('$\\chi_{c}$')
# ax.set_yscale('log')

# linestyles = ['solid', 'dotted', 'dashdot']
# i=0
# for mu in mus:

#     with open('Data/asymp_chi_crit_mu_' + str(mu) + '_eps_' + str(epsilon) + '_a0max_7.pkl', 'rb') as f:
#         results_dict = pickle.load(f)

#     chi_crits = results_dict['chi_crits']
#     a0s = results_dict['a0s']
#     print(a0s[1]-a0s[0])
#     print(min(a0s))
#     args = results_dict['mu_epsilon_stepsize_criterion']

#     ax.plot(a0s, chi_crits, label = f'$\\mu$ = {mu}', linestyle = linestyles[i])
#     ax.legend(loc = 'upper right')
#     i=i+1

    
##### Fig. 3.5

# def calculate_chi_crits_rbs(Psi,n_mu,n_eps):
#     epsilons = np.linspace(0.01, 0.5, n_eps)
    
#     mu_matrix = np.zeros((n_mu,n_eps))
#     chi_crits = np.zeros((n_mu,n_eps))
#     rbs = np.zeros((n_mu,n_eps))
    
#     j = 0
    
#     for epsilon in epsilons:

#         mus = np.linspace(0,4*np.pi*epsilon*Psi/9-1e-6,n_mu)
#         mu_matrix[:,j] = mus
        
#         i = 0
        
#         for mu in mus:
            
#             print(f'{Psi}_{epsilon}_{mu}')
            
#             chis, crit_chi, crit_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi)
            
#             chi_crits[i,j] = crit_chi
#             rbs[i,j] = rb
            
#             i = i+1            
#         j = j+1
        
#     results_dict = {'Psi':Psi, 'epsilons':epsilons,'mu_matrix':mu_matrix, 'crit_chis':chi_crits,'rbs':rbs }
    
#     fname = f'cmap_Psi_{Psi}_nmus_{n_mu}_neps_{n_eps}'
    
#     with open('Data/'+ fname + '.pkl', 'wb') as handle:
#         pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#     return 1

# def generate_cplots_parameter_exploration(Psis, nmus, neps):
    
#     fig, ax = plt.subplots(2,len(Psis))

#     for i,Psi in enumerate(Psis):
        
#         with open(f'Data/cmap_Psi_{Psi}_nmus_{nmus}_neps_{neps}.pkl', 'rb') as f:
#             data = pickle.load(f)
        
#         crit_chis = data['crit_chis']
#         rbs = data['rbs']
    
#         mu_matrix = data['mu_matrix']
    
#         epsilons = np.reshape(data['epsilons'], (1,neps))
#         epsilon_matrix = np.repeat(epsilons, nmus, axis = 0)
    
        
#         contour1 = ax[0,i].contourf(epsilon_matrix, mu_matrix, np.log10(crit_chis), levels = 100)
#         ax[0,i].set_xlabel('$\\epsilon$')
#         ax[0,i].set_ylabel('$\\mu$')
#         cbar1 = fig.colorbar(contour1, ax=ax[0,i])
#         cbar1.set_label('$\\log_{10}(\\chi_c)$')
#         ax[0,i].set_title(f'$\\Psi=$ {Psi}')
    
#         contour2 = ax[1,i].contourf(epsilon_matrix, mu_matrix, np.log10(rbs), levels = 100)
#         ax[1,i].set_xlabel('$\\epsilon$')
#         ax[1,i].set_ylabel('$\\mu$')
#         cbar2 = fig.colorbar(contour2, ax=ax[1,i])
#         cbar2.set_label('$\\log_{10}(r_B)$')
#         ax[1,i].set_title(f'$\\Psi=$ {Psi}')
    
#     return 0

# Psis = [3,5,7]
# n_mu = 500
# n_eps = 500

# for Psi in Psis:
#     calculate_chi_crits_rbs(Psi,n_mu,n_eps)

# generate_cplots_parameter_exploration(Psis, n_mu, n_eps)

##### Fig. 3.6

# def generate_contour_plot_asymp(model, axlim, n_angle, epsilon, mu_frac, Psi, axis,rb):
    
#     mu = model.mu
#     epsilon = model.epsilon
#     Psi = model.Psi
    
#     theta_grid = np.linspace(0,2*np.pi,n_angle)
#     rhat = model.rhat
#     axlim = axlim/rb
#     psi = np.zeros((len(theta_grid),len(rhat)))
    
#     i = 0
#     for theta in theta_grid:
#         psi[i,:] = model.psi_theta_r(theta)
#         i = i+1
#     R,THETA = np.meshgrid(rhat/rb,theta_grid)
    
#     X = R*np.sin(THETA)
#     Y = R*np.cos(THETA)
    
    
#     axis.contour(X, Y, psi, levels = np.linspace(0,np.max(psi),20), colors = 'k', linewidths = 1, vmin = 0)
#     axis.set_xlabel('$\\hat{r}/r_B$')
#     axis.set_ylabel('$\\hat{z}/r_B$')
#     axis.set_xlim(-axlim,axlim)
#     axis.set_ylim(-axlim,axlim)
#     axis.set_aspect('equal')
#     axis.set_title(f'({Psi}, {mu_frac}, {epsilon})')

        
#     return 1

# epsilon = 0.1
# Psis = [3,5]
# mu_crits = [0.4030162313714636,0.6907869873507858]
# mus_frac = [0,0.3,1]

# # epsilon = 0.1
# # Psis = [5]
# # mu_crits = [0.6907869873507858]
# # mus_frac = [0.3]


# nrows, ncols = len(Psis), len(mus_frac)
# dx, dy = 1, 1
# figsize = plt.figaspect(float(dy * nrows) / float(dx * ncols))
# fig4, axes = plt.subplots(nrows,ncols, figsize = figsize)

# for j in range(len(Psis)):
    
#     Psi = Psis[j]
#     mu_crit = mu_crits[j]
    
#     for i in range(len(mus_frac)):
        
#         mu = mus_frac[i]*mu_crit
        
#         if(nrows==1 and ncols==1):
#             axis = axes
#         else:
#             axis = axes[j,i] 
        
#         chis, crit_chi, crit_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi)
#         generate_contour_plot_asymp(crit_model, rb, 1000, epsilon, mus_frac[i], Psi, axis,rb)

##### Fig. 3.7

# def zero_crossing(f,r):

#     first_negative = np.where(f<0)[0][0]
#     indices_for_interpolation = range(max(0,first_negative-10), min(len(f), first_negative + 10))
#     zero_crossing = np.interp(0,f[indices_for_interpolation][::-1],r[indices_for_interpolation][::-1])
    
#     return zero_crossing

# def determine_eccentricity_profile(equatorial_profile, polar_profile, r_grid, npoints):
    
#     ahats = np.linspace(0.1, zero_crossing(equatorial_profile,r_grid) - 0.1, npoints)
#     bhats = np.zeros(len(ahats))
#     e  = np.zeros(len(bhats))

#     for i in range(len(bhats)):
#         ahat = ahats[i]
#         psi = np.interp(ahat, r_grid, equatorial_profile)
#         bhats[i] = np.interp(psi, polar_profile[::-1], r_grid[::-1])
        
#     e = np.sqrt(1-bhats**2/ahats**2)
    
    
#     return e, bhats, ahats

# epsilon = 0.1
# Psi = 5
# mu_crit = 0.6907869873507858
# mus_frac = [0,0.3,1]
# linestyles = ['solid', 'dotted', 'dashdot']
# fig4, axis = plt.subplots(1,1)

# for i in range(len(mus_frac)):
    
#     mu = mus_frac[i]*mu_crit
#     #axis = axes[i] 
    
#     chis, crit_chi, crit_model,rb = determine_critical_chi_asymp(mu,epsilon, Psi)
#     polar_psi = crit_model.psi_theta_r(0)
#     equatorial_psi = crit_model.psi_theta_r(np.pi/2)
#     rhat = crit_model.rhat
#     e1, bhats1, ahats1 = determine_eccentricity_profile(equatorial_psi, polar_psi, rhat, 100)

#     axis.set_title(f'$(\\Psi,\\epsilon)$ = ({Psi},{epsilon})')
#     axis.plot(ahats1[1:]/rb, e1[1:], label = f'$\\mu/\\mu_c$ = {mus_frac[i]}', linestyle = linestyles[i])
#     axis.set_xlabel('$\\hat{a}/r_B$')
#     axis.set_ylabel('e')
#     axis.legend(loc = 'upper left')

##### Fig. 3.8

# def asymp_M_hat(model):
    
#     M_hat = (-4*np.pi*model.rho_hat(model.Psi))/9 * ( (9*mu)/(4*np.pi) + model.lambda_0 + model.chi * model.lambda_1)
    
#     return M_hat

# def combined_beta_E(model, n_theta, n_rhat):
    
#     Psi = model.Psi
#     epsilon = model.epsilon
#     chi = model.chi
#     mu = model.mu

#     rhat = np.logspace(np.log10(epsilon), np.log10(max(model.rhat)), n_rhat)
#     theta_grid = np.linspace(0,np.pi,n_theta)
    
#     integrand2_2d = np.zeros((n_theta, n_rhat))
    
#     for i, theta in enumerate(theta_grid):

#         psi = np.interp(x = rhat, xp = model.rhat, fp = model.psi_theta_r(theta), left = 0, right = 0)
        
#         rho_hat = np.array(list(map(model.rho_hat,psi)))

#         M_hat = asymp_M_hat(model)
        
#         integrand2_2d[i,:] = rho_hat * (psi - (9/2) * chi * rhat**2 * np.sin(theta)**2 - (9 * mu)/(4 * np.pi * rhat)  ) * rhat**2 * np.sin(theta)    
#         integral2 = np.pi * simpson(y = simpson(y = integrand2_2d, x = rhat, axis = -1), x = theta_grid, axis = 0)
        
#     W_hat = (model.alpha_0 + chi*model.alpha_1) * 0.5 * M_hat  - integral2
    
#     beta = ((81 * 3**(2/3))/(16 * np.pi**2 * model.rho_hat(Psi)**2)) * M_hat**(4/3)
    
#     E = ((-8 * np.pi**2 * model.rho_hat(Psi)**2) / (9*3**(8/3))) * (W_hat/(M_hat**(7/3))) 
    
    
#     return beta, E

# # First generate the usual LoKi spiral
# chi = 0
# epsilon = 0.1
# mu = 0.3

# n_theta = 100
# n_rhat = 1000

# a0s = np.concatenate((np.linspace(0.05, 0.2, 600), np.linspace(0.2, 25, 600)))

# betas_base = np.zeros(len(a0s))
# Es_base = np.zeros(len(a0s))

# for i, a0 in enumerate(a0s):
    
#     Psi = a0 + (9*mu)/(4*np.pi * epsilon)
    
#     model = RLoKi_asymp(mu, epsilon, Psi, chi)
#     betas_base[i], Es_base[i] =  combined_beta_E(model, n_theta, n_rhat)
#     print(i)

# fig, ax = plt.subplots(1,1)
# ax.plot(Es_base, betas_base, label = '$\\chi$ = 0')#, marker = 'x')
# ax.set_xlabel('$\\mathcal{E}$')
# ax.set_ylabel('$\\beta$')
# ax.set_xlim(0,2)
# ax.set_ylim(0,1.7)

# ### A track for fixed chi, varying a0

# mu = 0.3
# chi = 1e-5
# epsilon = 0.1

# a0s = np.concatenate((np.linspace(0.01, 0.1, 250), np.linspace(0.1, 0.2, 250), np.linspace(0.2, 30, 700)))

# betas = np.zeros(len(a0s))
# Es = np.zeros(len(a0s))

# for i, a0 in enumerate(a0s):
    
#     Psi = a0 + (9*mu)/(4*np.pi * epsilon)
    
#     _, crit_chi, _,_ = determine_critical_chi_asymp(mu,epsilon, Psi)
    
#     if (chi<= crit_chi):
        
#         model = RLoKi_asymp(mu, epsilon, Psi, chi)
#         betas[i], Es[i] =  combined_beta_E(model, n_theta, n_rhat)
        
#     else:
#         print('Oooh, a gap')
#         betas[i], Es[i] =  np.NaN, np.NaN
        
#     print(i)
    
# ax.plot(Es, betas, label = f'$\\chi$ = {chi}')#, marker = 'x')
# axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4], xlim=(0.12,0.42 ), ylim=(1.2, 1.35))
# axins.plot(Es_base, betas_base, label = '$\\chi$ = 0')
# axins.plot(Es, betas, label = f'$\\chi$ = {chi}')
# ax.legend(loc = 'lower right')
# ax.indicate_inset_zoom(axins, edgecolor="black")


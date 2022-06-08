import numpy as np
import scipy.linalg as la
import dolfin as dl
import code
import sys

class sb_linear_cost():
	def __init__(self,prior_mean,prior_cov,pde_solver,obs_operator,noise_cov,obs_real):
		self.prior_cov = prior_cov
		self.pde_solver = pde_solver
		self.obs_operator = obs_operator
		self.noise_cov = noise_cov
		self.obs_real = obs_real
		self.prior_mean = prior_mean

	def evaluate(self,u):
		# Evaluate cost function
		forward = self.obs_operator.apply_obs(self.pde_solver.F_mult(u))
		misfit = forward - self.obs_real
		prior_diff = u.copy()
		prior_diff.set_local(u.get_local() - self.prior_mean.get_local())
		s_mis = self.noise_cov.apply_noise_cov_inv(misfit)
		J_mis = 0.5*np.dot(misfit,s_mis)
		s_reg = self.prior_cov.mult_inv_full(prior_diff)
		J_reg = 0.5*np.dot(prior_diff.get_local(),s_reg.get_local())
#		J = J_mis + J_reg

		# Evaluate gradient
		G_mis = self.pde_solver.Ft_mult(self.obs_operator.apply_obs_transpose(s_mis))
		G = u.copy()
		G.set_local(G_mis.get_local()+s_reg.get_local())
		return J_mis,J_reg,G

	

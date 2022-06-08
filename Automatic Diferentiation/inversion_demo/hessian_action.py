import numpy as np
import code
import scipy.linalg as la
import dolfin as dl
import scipy.sparse.linalg as spla
import randomEigensolver

class reduced_Hessian():
    def __init__(self,V,rank,p,prior,pde_solver,obs_operator,noise_cov):
        self.prior = prior
        self.pde_solver = pde_solver
        self.obs_operator = obs_operator
        self.noise_cov = noise_cov
        self.rank = rank
        self.p = p
        self.V = V
        nDof = V.dim()  

        temp_vec = dl.Function(V).vector()
        def prior_func(x):
            temp_vec.set_local(x)
            y = self.prior.mult_full(temp_vec)
            return y.get_local()

        prior_op = spla.LinearOperator((V.dim(),V.dim()),matvec=prior_func)

        def prior_inv_func(x):
            temp_vec.set_local(x)
            y = self.prior.mult_inv_full(temp_vec)
            return y.get_local()

        prior_inv_op = spla.LinearOperator((V.dim(),V.dim()),matvec=prior_inv_func) 

        def H_mis_func(x):
            temp_vec.set_local(x)
            y = self.H_mis_full_mult(temp_vec)
            return y.get_local()
        H_mis_op = spla.LinearOperator((V.dim(),V.dim()),matvec=H_mis_func)

        d,U  = randomEigensolver.singlePassG(H_mis_op,prior_inv_op,prior_op,rank,p) 
        self.eigenvalues = d
        self.eigenvectors = U

    def H_sb_mult(self,x):
        # return the action of the low-rank Hessian on a vector
        t1 = self.prior.mult_inv_full(x)
        t2 = np.dot(self.eigenvectors.T,t1.get_local())
        t3 = np.multiply(self.eigenvalues,t2)
        t4 = np.dot(self.eigenvectors,t3)
        t5 = t4 + x.get_local()
        t5_pvec = x.copy()
        t5_pvec.set_local(t5)
        t6 = self.prior.mult_inv_full(t5_pvec)
        return t6
    
    def A_mult(self,x):
        t1 = self.prior.mult_inv_full(x)
        t2 = np.dot(self.eigenvectors.T,t1.get_local())
        t3 = np.multiply(-1.0*np.ones(t2.shape),t2)
        t4 = np.dot(self.eigenvectors,t3)
        t5 = t4 + x.get_local()
        t5_pvec = x.copy()
        t5_pvec.set_local(t5)
        t6 = self.prior.mult_inv_full(t5_pvec)
        return t6

    def H_sb_inv_mult(self,x):
        D = np.multiply(self.eigenvalues,np.power(self.eigenvalues+1.0,-1.0))
        t1 = np.dot(self.eigenvectors.T,x.get_local())
        t2 = np.multiply(D,t1)
        t3 = np.dot(self.eigenvectors,t2)
        t4 = self.prior.mult_full(x)
        t5 = t4.get_local()-t3
        t5_pvec = x.copy()
        t5_pvec.set_local(t5)
        return t5_pvec

    def H_mis_full_mult(self,x):
        # compute F*(previous term)
        t2 = self.pde_solver.F_mult(x)
        # compute B*(previous)
        t3 = self.obs_operator.apply_obs(t2)
        # compute (Sigma_noise^-1)*(previous)
        t4 = self.noise_cov.apply_noise_cov_inv(t3)
        # compute B^T*(previous)
        t5 = self.obs_operator.apply_obs_transpose(t4)
        # compute F^T*(previous)
        t6 = self.pde_solver.Ft_mult(t5)
        return t6

class full_Hessian_sb():
    def __init__(self,prior,pde_solver,obs_operator,noise_cov):
        self.prior = prior
        self.pde_solver = pde_solver
        self.obs_operator = obs_operator
        self.noise_cov = noise_cov

    def H_mult(self,x):
        t1 = self.prior.mult_inv_full(x)
        t2 = self.pde_solver.F_mult(x)
        t3 = self.obs_operator.apply_obs(t2)
        t4 = self.noise_cov.apply_noise_cov_inv(t3)
        t5 = self.obs_operator.apply_obs_transpose(t4)
        t6 = self.pde_solver.Ft_mult(t5)
        t7 = x.copy()
        t7.set_local(t1.get_local()+t6.get_local())
        return t7
    

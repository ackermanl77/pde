
import dolfin as dl
import numpy as np
from scipy.sparse import lil_matrix
import code
import sys


class Adv_diff_solver():
    def __init__(self,mesh,V,wind_velocity,vel_space,dt,kappa,T_final):
        self.dt = dt
        self.V = V
        self.T_final = T_final
        h = dl.CellDiameter(mesh)
        u = dl.TrialFunction(V)
        v = dl.TestFunction(V)
        kappa_expr = dl.Constant(kappa)
        dt_expr = dl.Constant(dt)
        
        r_trial = u + dt_expr*( -dl.div(kappa*dl.nabla_grad(u))+ dl.inner(wind_velocity, dl.nabla_grad(u)) )
        r_test  = v + dt_expr*( -dl.div(kappa*dl.nabla_grad(v))+ dl.inner(wind_velocity, dl.nabla_grad(v)) )
        vnorm = dl.sqrt(dl.inner(wind_velocity, wind_velocity))
        tau = (h*h)/(dl.Constant(2.)*kappa)

        self.M = dl.assemble( dl.inner(u,v)*dl.dx )
        self.M_stab = dl.assemble( dl.inner(u, v+tau*r_test)*dl.dx )
        self.Mt_stab = dl.assemble( dl.inner(u+tau*r_trial,v)*dl.dx )
        Nvarf  = (dl.inner(kappa *dl.nabla_grad(u), dl.nabla_grad(v)) + dl.inner(wind_velocity, dl.nabla_grad(u))*v )*dl.dx
        Ntvarf  = (dl.inner(kappa *dl.nabla_grad(v), dl.nabla_grad(u)) + dl.inner(wind_velocity, dl.nabla_grad(v))*u )*dl.dx
        self.N  = dl.assemble( Nvarf )
        self.Nt = dl.assemble(Ntvarf)
        stab = dl.assemble( tau*dl.inner(r_trial, r_test)*dl.dx)
        self.L = self.M + dt*self.N + stab
        self.Lt = self.M + dt*self.Nt + stab
        
        self.solver = dl.PETScKrylovSolver("gmres", "ilu")
        self.solver.set_operator(self.L)
        self.solvert = dl.PETScKrylovSolver("gmres", "ilu")
        self.solvert.set_operator(self.Lt)
        self.forward_count = 0

    def forward_step(self,u0):
        y = u0.copy()
        z = u0.copy()
        self.M_stab.mult(u0,y)
        self.solver.solve(z,y)
        return z
        
    def forward_t_step(self,u0):
        y = u0.copy()
        z = u0.copy()
        self.solvert.solve(y,u0)
        self.Mt_stab.mult(y,z)
        return z

    def F_mult(self,u0,output_file="none"):
        self.forward_count += 1        
        t = 0.0
        u_old = dl.Function(self.V)
        u_old.vector().set_local(u0.get_local())
        u_new = dl.Function(self.V)
        if (str(output_file) != "none"):
            output_file << (u_old,t)
        while (t < self.T_final):
            u_new.vector().set_local(self.forward_step(u_old.vector()).get_local())
            t = t + self.dt
            u_old.vector().set_local(u_new.vector().get_local())
            if (str(output_file) != "none"):    
                output_file << (u_old,t)
        return u_new.vector()

    def Ft_mult(self,u0,output_file="none"):
        self.forward_count += 1
        t = 0.0
        u_old = dl.Function(self.V)
        u_old.vector().set_local(u0.get_local())
        u_new = dl.Function(self.V)
        if (str(output_file) != "none"):
            output_file << (u_old,t)
        while (t < self.T_final):
            u_new.vector().set_local(self.forward_t_step(u_old.vector()).get_local())
            t = t + self.dt
            u_old.vector().set_local(u_new.vector().get_local())
            if (str(output_file) != "none"):
                output_file << (u_old,t)
        return u_new.vector()

class Observation_operator():

    def __init__(self,nObs,mesh,V,solver):
#        np.random.seed(1)
        self.nObs = nObs
        self.nVert = mesh.num_vertices()

#       non_Dirichlet_vert_ind = np.array([v.index() for v in dl.vertices(mesh) if solver.interior.inside(v.point())])      

        non_Dirichlet_vert_ind = np.array([v.index() for v in dl.vertices(mesh)])
        dofmap = V.dofmap()
        nvertices = mesh.ufl_cell().num_vertices()
        indices = [dofmap.tabulate_entity_dofs(0, i)[0] for i in range(nvertices)]
        vertex_2_dof = dict()
        [vertex_2_dof.update(dict(vd for vd in zip(cell.entities(0),dofmap.cell_dofs(cell.index())[indices]))) for cell in dl.cells(mesh)]
        random_indices = np.random.choice(non_Dirichlet_vert_ind,self.nObs,replace=False)
        obs_vert_ind = random_indices
        obs_dof_ind = np.zeros(nObs)
        for i in range(0,nObs):
            vert_ind = obs_vert_ind[i]
            obs_dof_ind[i] = vertex_2_dof.get(vert_ind)

        self.obs_vert_coor = mesh.coordinates()[obs_vert_ind]
        self.B = lil_matrix((self.nObs,V.dim()))
        for i in range(0,nObs):
            self.B[i,obs_dof_ind[i]] = 1.0  
        self.Bt = self.B.transpose()
        self.V = V

    def apply_obs(self,u):
        d = self.B.dot(u.get_local())
        return d

    def apply_obs_transpose(self,d):
        u = dl.Function(self.V)
        u.vector().set_local(self.Bt.dot(d))
        return u.vector()


class Noise_covariance():

    def __init__(self,sigma_vec):
        self.sigma_vec = sigma_vec

    def apply_noise_cov(self,x):
        y = np.multiply(np.power(self.sigma_vec,2.0),x)
        return y

    def apply_noise_cov_inv(self,y):
        x = np.multiply(np.power(self.sigma_vec,-2.0),y)
        return x

class BiLaplacian_prior():
# Constructs prior covariance
    def __init__(self,Vh,gamma,delta):
        # Initialize prior and discretize components
        trial = dl.TrialFunction(Vh)
        test = dl.TestFunction(Vh)

        A_varf = dl.inner(dl.nabla_grad(trial),dl.nabla_grad(test))*dl.dx
        M_varf = dl.inner(trial,test)*dl.dx
        self.K = dl.assemble(dl.Constant(gamma)*A_varf+dl.Constant(delta)*M_varf)
        self.M = dl.assemble(M_varf)
        
        self.M_solver = dl.LUSolver()
#        self.M_solver.parameters["relative_tolerance"] = 1e-12
        self.M_solver.set_operator(self.M)
#        self.M_solver.parameters["error_on_nonconvergence"] = True
#        self.M_solver.parameters["nonzero_initial_guess"] = False

        self.K_solver = dl.LUSolver()
#        self.K_solver.parameters["relative_tolerance"] = 1e-12
        self.K_solver.set_operator(self.K)
#        self.K_solver.parameters["error_on_nonconvergence"] = True
#        self.K_solver.parameters["nonzero_initial_guess"] = False

    def mult_neg_half(self,x):
        # Operator to perform matrix-vector products with the prior
        q = x.copy()
        y = x.copy()
        self.K.mult(x,q)
        self.M_solver.solve(y,q)
        return y

    def mult_half(self,y):
        q = y.copy()
        x = y.copy()
        self.M.mult(y,q)
        self.K_solver.solve(x,q)
        return x
    
    def mult_full(self,y):
        q = y.copy()
        x = y.copy()
        p = y.copy()
        w = y.copy()
        self.K_solver.solve(x,y)
        self.M.mult(x,p)
        self.K_solver.solve(w,p)
        return w

    def mult_inv_full(self,y):
        x = y.copy()
        q = y.copy()
        p = y.copy()
        self.K.mult(y,x)
        self.M_solver.solve(p,x)
        self.K.mult(p,q)
        return q

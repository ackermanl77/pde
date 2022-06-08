"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 500*exp(-((x-0.5)^2 + (y-0.5)^2)/0.02)

and boundary conditions given by

    u(x, y)     = 0
    du/dn(x, y) = 0

using a discontinuous Galerkin formulation (interior penalty method).
"""

__author__    = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__      = "2007-10-02 -- 2008-12-19"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__   = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(24, 24)
V = FunctionSpace(mesh, "DG", 1)

# Define test and trial functions
u = TrialFunction(V)
v = TestFunction(V)

# Define normal component, mesh size and right-hand side
n = FacetNormal(mesh)
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2
f = Expression("500.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", V = V)

# Define parameters
alpha = 4.0
gamma = 8.0

# Define bilinear form
a = dot(grad(v), grad(u))*dx \
   - dot(avg(grad(v)), jump(u, n))*dS \
   - dot(jump(v, n), avg(grad(u)))*dS \
   + alpha/h_avg*dot(jump(v, n), jump(u, n))*dS \
   - dot(grad(v), u*n)*ds \
   - dot(v*n, grad(u))*ds \
   + gamma/h*v*u*ds

# Define linear form
L = v*f*dx

# Compute solution
problem = VariationalProblem(a, L)
u = problem.solve()

# Project solution to piecewise linears
P1 = FunctionSpace(mesh, "CG", 1)
u_proj = project(u, P1)

# Save solution to file
file = File("poisson.pvd")
file << u_proj

# Plot solution
plot(u_proj, interactive=True)


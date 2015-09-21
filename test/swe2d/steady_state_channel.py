# Tuomas Karna 2015-03-03
from scipy.interpolate import interp1d
from cofs import *
import math

Lx=5e3
Ly=1e3
# we don't expect converge as the reference solution neglects the advection term
mesh2d = RectangleMesh(5,1,Lx,Ly)

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name="bathymetry")
bathymetry2d.assign(100.0)

N = 20 # number of timesteps
dt = 100.
g = physical_constants['g_grav'].dat.data[0]
f = g/Lx # linear friction coef.

# --- create solver ---
solverObj = solver.flowSolver2d(mesh2d, bathymetry2d, order=1)
solverObj.nonlin = False
solverObj.TExport = dt
solverObj.T = N*dt
solverObj.timeStepperType = 'CrankNicolson'
solverObj.lin_drag = f
solverObj.dt = dt

# boundary conditions
inflow_tag = 1
outflow_tag = 2
inflow_func=Function(P1_2d)
inflow_func.interpolate(Expression(-1.0))
inflow_bc = {'un': inflow_func}
outflow_func=Function(P1_2d)
outflow_func.interpolate(Expression(0.0))
outflow_bc = {'elev': outflow_func}
solverObj.bnd_functions['shallow_water'] = {inflow_tag: inflow_bc, outflow_tag: outflow_bc}
parameters['quadrature_degree']=5

solverObj.assignInitialConditions(uv_init=Expression(("1.0","0.0")))
solver_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_package': 'mumps',
    'snes_monitor': False,
    'snes_type': 'newtonls'}
# reinitialize the timestepper so we can set our own solver parameters and gamma
# setting gamma to 1.0 converges faster to
solverObj.timeStepper = timeIntegrator.CrankNicolson(solverObj.eq_sw, solverObj.dt,
                                                                     solver_parameters, gamma=1.0)

solverObj.iterate()

uv, eta = solverObj.solution2d.split()

eta_ana = interpolate(Expression("1-x[0]/{}".format(Lx)), P1_2d)
area = Lx*Ly
l2norm = assemble(pow(eta-eta_ana,2)*dx)
rel_err =  math.sqrt(l2norm/area)
print rel_err
assert(rel_err<1e-3)
print "PASSED"

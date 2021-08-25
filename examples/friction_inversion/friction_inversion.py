from thetis import *
from firedrake_adjoint import *
import h5py
import time as time_mod


class OptimisationProgress(object):
    """
    Class for stashing progress of the optimisation routine.
    """
    J_progress = []
    dJdm_progress = []
    J = 0
    dJdm = 0
    m = 0
    i = 0
    tic = None
    nb_grad_evals = 0
    nb_func_evals = 0

    def reset_counters(self):
        self.nb_grad_evals = 0
        self.nb_func_evals = 0

    def update(self, j, djdm, m):
        """
        To call whenever variables are updated.
        """
        self.J = j
        self.dJdm = djdm
        self.m = m

    def start_clock(self):
        self.tic = time_mod.clock()

    def update_progress(self, state=None):
        """
        To call after successful line searches.
        """
        toc = time_mod.clock()
        elapsed = '-' if self.tic is None else f'{toc - self.tic:.1f} s'
        self.tic = toc
        if state is not None:
            self.update(*state)
        djdm = norm(self.dJdm)
        self.J_progress.append(self.J)
        self.dJdm_progress.append(djdm)
        np.save("outputs/J_progress", self.J_progress)
        np.save("outputs/dJdm_progress", self.dJdm_progress)
        print_output(f"line search {self.i:2d}: "
                     f"J={self.J:.3e}, dJdm={djdm:.3e}, "
                     f"func_ev={self.nb_func_evals}, "
                     f"grad_ev={self.nb_grad_evals}, duration {elapsed}")
        self.i += 1
        self.reset_counters()


op = OptimisationProgress()

# Spatial discretisation
lx = 100e3
nx = 30
delta_x = lx/nx
ny = 2
ly = delta_x * ny
mesh2d = RectangleMesh(nx, ny, lx, ly)

t_end = 8 * 3600.
u_mag = Constant(6.0)
t_export = 600.
dt = 600.

# Bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
x, y = SpatialCoordinate(mesh2d)
depth_oce = 10.0
depth_riv = 5.0
bathymetry_2d.interpolate(depth_oce + (depth_riv - depth_oce)*x/lx)

# Control parameter: Manning friction coefficient
fs_control = P1_2d
# fs_control = FunctionSpace(mesh2d, 'R', 0)
manning = Function(fs_control).assign(1.0e-03)
c = Control(manning)

print(f'initial Manning coeff: {manning.dat.data.min()} .. {manning.dat.data.max()}')

# Create solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.manning_drag_coefficient = manning
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.horizontal_velocity_scale = u_mag
options.fields_to_export = []
options.fields_to_export_hdf5 = []
options.swe_timestepper_type = 'CrankNicolson'
if not hasattr(options.swe_timestepper_options, 'use_automatic_timestep'):
    options.timestep = dt
dtc = Constant(options.timestep)
solver_obj.create_equations()

# Create 0D mesh
xy = [[30e3, ly/2.], [80e3, ly/2]]
mesh0d = VertexOnlyMesh(mesh2d, xy)
P0_0d = FunctionSpace(mesh0d, 'DG', 0)
elev_o = Function(P0_0d, name='gauge obs')
elev_d = Function(P0_0d, name='gauge data')
misfit = elev_o - elev_d

# regularization term in const function
# grad(manning) is roughly manning/delta_x = O(1e-6)
# misfit is O(1)
# thus gamma grad must be < 1e12
gamma_grad = Constant(5.0)
reg_c_grad = gamma_grad * dot(grad(manning), grad(manning))

# Set initial condition for elevation, piecewise linear function
elev_init = Function(P1_2d)
elev_height = 6.0
elev_ramp_lx = 80e3
elev_init.interpolate(conditional(x < elev_ramp_lx,
                                  elev_height*(1 - x/elev_ramp_lx),
                                  0.0))
solver_obj.assign_initial_conditions(elev=elev_init, uv=Constant((1e-5, 0)))

# Load data as a list of 0D Functions
with h5py.File('outputs/diagnostic_stations.hdf5', 'r') as f:
    A = np.array(f['station_A'])
    B = np.array(f['station_B'])
data = []
for a, b in zip(A, B):
    datum = Function(P0_0d)
    datum.dat.data[0] = a
    datum.dat.data[1] = b
    data.append(datum)


def qoi(t):
    """
    Compute square misfit between data and observations.
    """
    elev_o.interpolate(solver_obj.fields.solution_2d.split()[1])
    elev_d.assign(data.pop(0))
    area = lx*ly
    J_scale = 1e12
    J_misfit = assemble(dtc*misfit**2*dx)
    #op.J += J_misfit*J_scale/area
    J_reg_grag = assemble(dtc*reg_c_grad*dx)
    op.J += (J_misfit + J_reg_grag)*J_scale/area


def post_grad_cb(j, djdm, m):
    """
    Stash optimisation state.
    """
    op.update(j, djdm, m)
    op.nb_grad_evals += 1


def post_func_cb(*args):
    """
    Update func eval counter
    """
    op.nb_func_evals += 1


# Solve and setup reduced functional
solver_obj.iterate(update_forcings=qoi)
Jhat = ReducedFunctional(op.J, c, derivative_cb_post=post_grad_cb, eval_cb_post=post_func_cb)
stop_annotating()

# Consistency test
#print_output("Running consistency test")
#J = Jhat(manning)
#assert np.isclose(J, op.J)
#print_output("Consistency test passed!")

# Taylor test
#dc = Function(fs_control)
#dc.assign(1.0e-04)
#minconv = taylor_test(Jhat, manning, dc)
#assert minconv > 1.9
#print_output("Taylor test passed!")

# VTU outputs for optimisation progress
outfile_m = File("outputs/manning_progress.pvd")
outfile_dJdm = File("outputs/gradient_progress.pvd")


def cb(m):
    """
    Stash optimisation progress after successful line search.
    """
    op.update_progress()
    op.m.rename("Manning coefficient")
    op.dJdm.rename("Gradient")
    outfile_m.write(op.m)
    outfile_dJdm.write(op.dJdm)


# Run inversion
opt_method = "L-BFGS-B"
options = {
    'maxiter': 100,
    'ftol': 5e-3,
    'iprint': 101,
}
print_output(f"Running {opt_method} optimization")
op.reset_counters()
op.start_clock()
J = Jhat(manning)
op.update_progress(state=[float(J), Jhat.derivative(), manning])
manning_opt = minimize(Jhat, method=opt_method, bounds=[1e-4, 1e-1], callback=cb,
                       options=options)
print(f'Optimal Manning coeff: {manning_opt.dat.data.min()} .. {manning_opt.dat.data.max()}')
File("outputs/manning_optimised.pvd").write(manning_opt)

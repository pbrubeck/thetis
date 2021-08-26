from thetis import *
from firedrake_adjoint import *
import h5py
import time as time_mod
from scipy.interpolate import interp1d


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
    control_name = 'Manning coefficient'

    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.outfile_m = File(f'{self.output_dir}/control_progress.pvd')
        self.outfile_dJdm = File(f'{self.output_dir}/gradient_progress.pvd')

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
        np.save(f"{self.output_dir}/J_progress", self.J_progress)
        np.save(f"{self.output_dir}/dJdm_progress", self.dJdm_progress)
        print_output(f"line search {self.i:2d}: "
                     f"J={self.J:.3e}, dJdm={djdm:.3e}, "
                     f"func_ev={self.nb_func_evals}, "
                     f"grad_ev={self.nb_grad_evals}, duration {elapsed}")
        self.i += 1
        self.reset_counters()

        self.m.rename(self.control_name)
        self.dJdm.rename("Gradient")
        self.outfile_m.write(self.m)
        self.outfile_dJdm.write(self.dJdm)


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

# Set initial condition for elevation, piecewise linear function
elev_init = Function(P1_2d)
elev_height = 6.0
elev_ramp_lx = 80e3
elev_init.interpolate(conditional(x < elev_ramp_lx,
                                  elev_height*(1 - x/elev_ramp_lx),
                                  0.0))
solver_obj.assign_initial_conditions(elev=elev_init, uv=Constant((1e-5, 0)))

# Load observation time series
obs_dir = 'outputs_forward'
file_list = [
    f'{obs_dir}/diagnostic_timeseries_stationA_elev.hdf5',
    f'{obs_dir}/diagnostic_timeseries_stationB_elev.hdf5',
]
station_coords = []
station_time = []
station_vals = []
for f in file_list:
    with h5py.File(f) as h5file:
        t = h5file['time'][:].flatten()
        v = h5file['elev'][:].flatten()
        x = h5file.attrs['x']
        y = h5file.attrs['y']
        station_coords.append((x, y))
        station_time.append(t)
        station_vals.append(v)

# Construct timeseries interpolator
station_interpolators = []
for t, v in zip(station_time, station_vals):
    ip = interp1d(t, v)
    station_interpolators.append(ip)


def interp_observations(t):
    return [float(ip(t)) for ip in station_interpolators]


# Create 0D mesh for station evaluation
mesh0d = VertexOnlyMesh(mesh2d, station_coords)
P0_0d = FunctionSpace(mesh0d, 'DG', 0)
elev_obs = Function(P0_0d, name='gauge observations')
elev_mod = Function(P0_0d, name='gauge modeled')
misfit = elev_obs - elev_mod

# regularization term in const function
# grad(manning) is roughly manning/delta_x = O(1e-6)
# misfit is O(1)
# thus gamma grad must be < 1e12
gamma_grad = Constant(5.0)
reg_c_grad = gamma_grad * dot(grad(manning), grad(manning))

obs_func_list = []  # need to keep target data in memory for annotation


def qoi():
    """
    Compute square misfit between data and observations.

    NOTE: this should be called as a post-solve callback operator
    """
    t = solver_obj.simulation_time
    obs_func = Function(P0_0d)
    obs_func.dat.data[:] = interp_observations(t)
    obs_func_list.append(obs_func)
    elev_obs.assign(obs_func)
    elev_mod.interpolate(solver_obj.fields.elev_2d)
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

solver_obj.iterate(export_func=qoi)
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


def cb(m):
    """
    Stash optimisation progress after successful line search.
    """
    op.update_progress()


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

from thetis import *

lx = 100e3
nx = 50
delta_x = lx/nx
ny = 2
ly = delta_x * ny
mesh2d = RectangleMesh(nx, ny, lx, ly)

t_end = 5 * 3600.
u_mag = Constant(6.0)
t_export = 300.
dt = 300.

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = 5*t_export

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
# assign bathymetry to a linear function
x, y = SpatialCoordinate(mesh2d)
depth_oce = 10.0
depth_riv = 5.0
bathymetry_2d.interpolate(depth_oce + (depth_riv - depth_oce)*x/lx)

# create solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.manning_drag_coefficient = Constant(1e-2)
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.horizontal_velocity_scale = u_mag
options.fields_to_export = ['uv_2d', 'elev_2d']
options.swe_timestepper_type = 'CrankNicolson'
if not hasattr(options.swe_timestepper_options, 'use_automatic_timestep'):
    options.timestep = dt

solver_obj.create_equations()
# store elevation time series at stations
xy = [[30e3, ly/2.], [80e3, ly/2]]
cb = DetectorsCallback(solver_obj, xy, ['elev_2d'], name='stations',
                       detector_names=['station_A', 'station_B'], append_to_log=False)
solver_obj.add_callback(cb)

# set initial condition for elevation, piecewise linear function
elev_init = Function(P1_2d)
elev_height = 6.0
elev_ramp_lx = 80e3
elev_init.interpolate(conditional(x < elev_ramp_lx,
                                  elev_height*(1 - x/elev_ramp_lx),
                                  0.0))
solver_obj.assign_initial_conditions(elev=elev_init, uv=Constant((1e-5, 0)))
solver_obj.iterate()

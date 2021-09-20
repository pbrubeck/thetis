"""
Generic methods for converting data between different spatial coordinate systems.
Uses pyproj library.
"""

import pyproj
import numpy


UTM_ZONE10 = pyproj.Proj(
    proj='utm',
    zone=10,
    datum='WGS84',
    units='m',
    errcheck=True)
LL_WGS84 = pyproj.Proj(proj='latlong', datum='WGS84', errcheck=True)


def convert_coords(source_sys, target_sys, x, y):
    """
    Converts coordinates from source_sys to target_sys

    This function extends pyproj.transform method by handling NaNs correctly.

    :arg source_sys: pyproj coordinate system where (x, y) are defined in
    :arg target_sys: target pyproj coordinate system
    :arg x: x coordinate
    :arg y: y coordinate
    :type x: float or numpy.array_like
    :type y: float or numpy.array_like
    """
    if isinstance(x, numpy.ndarray):
        # proj may give wrong results if nans in the arrays
        lon = numpy.full_like(x, numpy.nan)
        lat = numpy.full_like(y, numpy.nan)
        goodIx = numpy.logical_and(numpy.isfinite(x), numpy.isfinite(y))
        lon[goodIx], lat[goodIx] = pyproj.transform(
            source_sys, target_sys, x[goodIx], y[goodIx])
    else:
        lon, lat = pyproj.transform(source_sys, target_sys, x, y)
    return lon, lat


def get_vector_rotation_matrix(source_sys, target_sys, x, y, delta=None):
    """
    Estimate rotation matrix that converts vectors defined in source_sys to
    target_sys.

    Assume that we have a vector field defined in source_sys: vectors located at
    (x, y) define the x and y components. We can then rotate the vectors to
    represent x2 and y2 components of the target_sys by applying a local
    rotation:

    .. code-block:: python

        R, theta = get_vector_rotation_matrix(source_sys, target_sys, x, lat)
        v_xy = numpy.array([[v_x], [v_y]])
        v_new = numpy.matmul(R, v_xy)
        v_x2, v_y2 = v_new

    """
    if delta is None:
        delta = 1e-6  # ~1 m in LL_WGS84
    x1, y1 = pyproj.transform(source_sys, target_sys, x, y)

    x2, y2 = pyproj.transform(source_sys, target_sys, x, y + delta)
    dxdl = (x2 - x1) / delta
    dydl = (y2 - y1) / delta
    theta = numpy.arctan2(-dxdl, dydl)

    c = numpy.cos(theta)
    s = numpy.sin(theta)
    R = numpy.array([[c, -s], [s, c]])

    return R, theta


class VectorCoordSysRotation(object):
    """
    Rotates vectors defined in source_sys coordinates to a different coordinate
    system.

    """
    def __init__(self, source_sys, target_sys, x, y):
        """
        :arg source_sys: pyproj coordinate system where (x, y) are defined in
        :arg target_sys: target pyproj coordinate system
        :arg x: x coordinate
        :arg y: y coordinate
        """
        R, theta = get_vector_rotation_matrix(source_sys, target_sys, x, y)
        self.rotation_sin = numpy.sin(theta)
        self.rotation_cos = numpy.cos(theta)

    def __call__(self, v_x, v_y, i_node=None):
        """
        Rotate vectors defined by the `v_x` and `v_y` components.

        :arg v_x, v_y: vector x, y components
        :kwarg ix_node: If not None, rotate the i-th vector instead of the
            whole array
        """
        # | c -s | | v_x |
        # | s  c | | v_y |
        f = [i_node] if i_node is not None else slice(None, None, None)
        u = v_x * self.rotation_cos[f] - v_y * self.rotation_sin[f]
        v = v_x * self.rotation_sin[f] + v_y * self.rotation_cos[f]
        return u, v

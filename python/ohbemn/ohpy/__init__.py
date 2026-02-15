from . import geometry
from . import wave
from . import rcoeff
from .boundary import Region
from .solver import Solver

import numpy as np


def region_from_xy(x, y):
    """
    Return vertices and edges from a list of x and y coordinates in a polygon.

    This does not guarantee that the polygon points are given in a clockwise
    direction.
    """
    if x[-1] == x[0] and y[-1] == y[0]:
        # first and last point the same, these should be connected by the edges
        # map below.
        x = x[:-1]
        y = y[:-1]

    vertices = np.vstack((x, y)).T

    # set up edges
    e0 = np.arange(0, vertices.shape[0] - 1)
    e1 = e0 + 1

    # connect end with start
    e0 = np.append(e0, [e1[-1]])
    e1 = np.append(e1, [0])

    assert len(e0) == vertices.shape[0]

    edges = np.vstack((e0, e1)).T

    return vertices, edges


def region_from_polygon(polygon: 'shapely.Polygon'):
    """
    Return vertices and edges from a shapely polygon.
    """
    import shapely

    polygon = shapely.orient_polygons(polygon, exterior_cw=True)
    points = np.array(polygon.exterior.coords)
    return region_from_xy(points[:, 0], points[:, 1])

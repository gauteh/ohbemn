import io
import pandas as pd
from shapely import Polygon
import ohbemn as oh
from ohbemn import ohpy
import numpy as np


def test_square():
    v, e = ohpy.geometry.square()
    r = ohpy.Region(v, e)
    # print(r)
    print(r.len())

    print(v.shape, e.shape)
    r2 = oh.Region(v, e)
    print(r2)
    assert r2.len() == r.len()


def test_polygon():
    # fig 3.17 in Berkhoff (1976)
    points = pd.read_csv(
        io.StringIO("""
   0.5715,   0.9508
   0.5451,   0.8975
   0.1189,   0.8869
   0.0669,   0.8455
   0.0427,   0.7376
   0.0442,   0.5452
   0.0928,   0.5471
   0.0955,   0.3983
   0.0409,   0.3950
   0.0408,   0.1981
   0.0634,   0.0963
   0.1529,   0.0402
   0.6483,   0.0367
   0.7218,   0.0840
   0.7589,   0.1623
   0.8013,   0.3231
   0.7536,   0.3583
   0.7923,   0.4858
   0.8381,   0.4569
   0.9040,   0.6107
   0.9682,   0.8375
   0.9662,   0.9463
   0.5715,   0.9508
   """))

    polygon = Polygon(points.itertuples(index=False, name=None))
    print(polygon)

    v, e = oh.region_from_polygon(polygon)
    region = oh.Region(v, e)
    print(region)
    print(region.normals())

    # plt.figure()
    # region.plot(plt.gca())
    # plt.show()

    n = region.normals()

    assert n[0][1] > 0., "wrong orientation, normal should be outwards"
    assert ~np.any(
        np.isnan(n)
    ), "can happen with zero length segment, e.g. same point twice in coords"

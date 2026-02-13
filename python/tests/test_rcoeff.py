import pytest
import numpy as np

import ohbemn as oh
from ohbemn import ohpy
from ohbemn.ohpy import rcoeff
from ohbemn.ohpy.rcoeff import flipdir_deg, dir_from_to_incidence

def test_flipdir():
    # from -> to
    assert flipdir_deg(270) == 90
    assert flipdir_deg(360) == 180
    assert flipdir_deg(0) == 180

    assert flipdir_deg(90) == 270

    # to -> from
    assert flipdir_deg(270) == 90
    assert flipdir_deg(360) == 180
    assert flipdir_deg(0) == 180

    assert flipdir_deg(90) == 270

def test_normal():
    #%% testing a normal angle against a theta

    n_x = [1, 0] # vertical segment, in x dir
    n_y = [0, 1] # horizontal segment, normal in y dir

    # horizontal segment, normal pointing upwards 90 deg
    a = np.arctan2(n_y[1], n_y[0])
    assert a == np.pi/2, a

    # vertiacl segment, normal pointing right (0 deg)
    a = np.arctan2(n_x[1], n_x[0])
    assert a == 0, a

    # vertiacl segment, normal pointing left (180 deg)
    a = np.arctan2(n_x[1], -n_x[0])
    assert a == np.pi, a

    th = np.deg2rad(20)

    mth = rcoeff.theta_normal(th, n_y)
    np.testing.assert_allclose(mth, th)

    # normal pointing opposite dir of incidence angle.
    n = [np.cos(np.pi/2+th), np.sin(np.pi/2+th)]
    print(n)
    # n_x = [-np.sin(th), np.cos(th)]
    mth = rcoeff.theta_normal(th, n)
    print(np.rad2deg(mth))
    np.testing.assert_allclose(np.rad2deg(mth), 0.)

    # normal pointing 20 deg on other side of normal of incidence angle
    nth = np.deg2rad(90-20)
    n_x = [np.cos(nth), np.sin(nth)]
    mth = rcoeff.theta_normal(th, n_x)
    print(np.rad2deg(th), np.rad2deg(nth), n_x, np.rad2deg(mth))
    np.testing.assert_allclose(np.rad2deg(mth), 40.)

    #%% Test converting angles
    n_y = [0, 1]
    n_x = [1, 0]

    # from left
    # kx, ky, a = rcoeff.dir_from_k(270)
    # np.testing.assert_allclose(a, 360-270)

    # rotate: shouldn't rotate at all
    # ar = rcoeff.theta_normal(np.deg2rad(a), n_y)
    # print(np.rad2deg(ar))
    # assert np.rad2deg(ar) == 90.

    # rotate: should rotate -90 to 0.
    # ar = rcoeff.theta_normal(np.deg2rad(a), [-1, 0])
    # print(np.rad2deg(ar))
    # assert np.rad2deg(ar) == 0.

    # from left++
    # kx, ky, a = rcoeff.dir_from_k(274)
    # np.testing.assert_allclose(a, 360-274)

    # rotate: shouldn't rotate at all
    # ar = rcoeff.theta_normal(np.deg2rad(a), n_y)
    # print(np.rad2deg(ar))
    # assert np.rad2deg(ar) == 86.

    # rotate: should rotate -90 to 0.
    # ar = rcoeff.theta_normal(np.deg2rad(a), [-1, 0])
    # print(np.rad2deg(ar))
    # np.testing.assert_allclose(np.rad2deg(ar), -4.)
    # np.testing.assert_allclose(np.rad2deg(rcoeff.abs_incidence(ar)), 4.)

def test_signed_normal():
# horizontal segment (outside on top)
    n = [0, 1]  # normal pointing upwards: aligned with incident normal.

# some basic tests
    nt, os = dir_from_to_incidence(0, n)  # from the top
    assert nt == 0
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(270), n)  # from w
    np.testing.assert_allclose(nt, np.pi / 2)
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(90), n)  # from e
    np.testing.assert_allclose(nt, np.pi / 2)
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(180), n)  # from s
    np.testing.assert_allclose(nt, 0)
    assert os == False

    nt, os = dir_from_to_incidence(np.deg2rad(135), n)  # from se
    np.testing.assert_allclose(nt, np.deg2rad(45))
    assert os == False

    nt, os = dir_from_to_incidence(np.deg2rad(180+45), n)  # from sw
    np.testing.assert_allclose(nt, np.deg2rad(45))
    assert os == False

    nt, os = dir_from_to_incidence(np.deg2rad(270+45), n)  # from nw
    np.testing.assert_allclose(nt, np.deg2rad(45))
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(0+45), n)  # from ne
    np.testing.assert_allclose(nt, np.deg2rad(45))
    assert os == True

# incident theta is symmetrical around normal.
# theta should be unchanged, since normal is pointing straight up.
    th = np.arange(270, 360)
    ths = np.arange(90, 0, -1)
    ntt = np.arange(90, 0, -1)

    assert len(th) == len(ths)

    for t, ts, ns in zip(th, ths, ntt):
        t = np.deg2rad(t)
        ts = np.deg2rad(ts)
        ns = np.deg2rad(ns)

        nt, os = dir_from_to_incidence(t, n)
        nts, os = dir_from_to_incidence(ts, n)

        np.testing.assert_allclose(nt,
                                ns,
                                err_msg=f"{np.rad2deg(nt)} != {np.rad2deg(ns)}")
        np.testing.assert_allclose(
            nts, ns, err_msg=f"{np.rad2deg(nts)} != {np.rad2deg(ns)}")

        assert os, "should be on outside"


## Normal pointing 45 deg to NW
# horizontal segment (outside on top)
    n = [-np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))]  # normal pointing upwards: aligned with incident normal.

# some basic tests
    nt, os = dir_from_to_incidence(0, n)  # from the top
    assert nt == np.deg2rad(45)
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(270), n)  # from w
    np.testing.assert_allclose(nt, np.deg2rad(45))
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(90), n)  # from e
    np.testing.assert_allclose(nt, np.deg2rad(45))
    assert os == False

    nt, os = dir_from_to_incidence(np.deg2rad(180), n)  # from s
    np.testing.assert_allclose(nt, np.deg2rad(45))
    assert os == False

    nt, os = dir_from_to_incidence(np.deg2rad(135), n)  # from se
    np.testing.assert_allclose(nt, np.deg2rad(0))
    assert os == False

    nt, os = dir_from_to_incidence(np.deg2rad(180+45), n)  # from sw
    np.testing.assert_allclose(nt, np.deg2rad(90))
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(270+45), n)  # from nw
    np.testing.assert_allclose(nt, np.deg2rad(0))
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(0+45), n)  # from ne
    np.testing.assert_allclose(nt, np.deg2rad(90))
    assert os == True


## Normal pointing 45 deg to NE
# horizontal segment (outside on top)
    n = [np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))]  # normal pointing upwards: aligned with incident normal.

# some basic tests
    nt, os = dir_from_to_incidence(0, n)  # from the top
    assert nt == np.deg2rad(45)
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(270), n)  # from w
    np.testing.assert_allclose(nt, np.deg2rad(45))
    assert os == False

    nt, os = dir_from_to_incidence(np.deg2rad(90), n)  # from e
    np.testing.assert_allclose(nt, np.deg2rad(45))
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(180), n)  # from s
    np.testing.assert_allclose(nt, np.deg2rad(45))
    assert os == False

    nt, os = dir_from_to_incidence(np.deg2rad(135), n)  # from se
    np.testing.assert_allclose(nt, np.deg2rad(90))
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(180+45), n)  # from sw
    np.testing.assert_allclose(nt, np.deg2rad(0))
    assert os == False

    nt, os = dir_from_to_incidence(np.deg2rad(270+45), n)  # from nw
    np.testing.assert_allclose(nt, np.deg2rad(90))
    assert os == True

    nt, os = dir_from_to_incidence(np.deg2rad(0+45), n)  # from ne
    np.testing.assert_allclose(nt, np.deg2rad(0))
    assert os == True

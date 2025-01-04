import ohbemn as oh
from ohbemn import ohpy


def test_square():
    v, e = ohpy.geometry.square()
    r = ohpy.Region(v, e)
    print(r)
    print(r.len())

    print(v.shape, e.shape)
    r2 = oh.Region(v, e)
    print(r2)
    assert r2.len() == r.len()

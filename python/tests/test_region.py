import ohbemn
from ohbemn.boundary import Region
from ohbemn.geometry import square

def test_square():
    v, e = square()
    r = Region(v, e)
    print(r)

    print(r.len())

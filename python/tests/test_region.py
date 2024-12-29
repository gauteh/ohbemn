import ohbemn
from ohbemn.boundary import Region
from ohbemn.geometry import Square

def test_square():
    v, e = Square()
    r = Region(v, e)
    print(r)

    print(r.len())

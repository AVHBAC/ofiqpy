"""Pure-function unit tests — no models required (runnable in CI)."""

from ofiqpy.measures.helpers import c_round, get_distance, get_middle
from ofiqpy.output import OFIQ_ORDER, header, row
from ofiqpy.sigmoid import STRUCT_DEFAULTS, _round_half_away, scalar_conversion


def test_version():
    import ofiqpy
    assert ofiqpy.__version__


def test_round_half_away():
    assert _round_half_away(0.5) == 1
    assert _round_half_away(1.5) == 2
    assert _round_half_away(2.5) == 3   # NOT banker's rounding (would be 2)
    assert _round_half_away(-0.5) == -1
    assert c_round(2.5) == 3


def test_sigmoid_clamps_and_shape():
    # struct defaults exist
    assert STRUCT_DEFAULTS["h"] == 100
    # clamps to [0, 100]
    assert scalar_conversion(1e9, h=100, x0=23, w=2.6) == 100
    assert scalar_conversion(-1e9, h=100, x0=23, w=2.6) == 0
    # midpoint of a plain sigmoid (a=0,s=1) at x==x0 is h/2
    assert scalar_conversion(23.0, h=100, x0=23.0, w=2.6) == 50


def test_geometry_helpers():
    assert get_distance((0, 0), (3, 4)) == 5.0
    assert get_middle([(0, 0), (2, 2)]) == (1, 1)
    assert get_middle([(0, 0), (1, 1)]) == (1, 1)  # round(0.5) half-away


def test_output_format():
    cols = header().split(";")
    # Filename + 28 raw + 28 scalar + time
    assert cols[0] == "Filename"
    assert cols[-1] == "assessment_time_in_ms"
    assert len(cols) == 1 + 2 * len(OFIQ_ORDER) + 1
    assert len(OFIQ_ORDER) == 28

    r = row("x/face.jpg", {"HeadSize": (0.3, 42)}).split(";")
    assert r[0] == "face.jpg"
    # unimplemented components -> FailureToAssess sentinel (scalar -1)
    hs_i = 1 + OFIQ_ORDER.index("HeadSize")
    assert r[hs_i] == "0.300000"


def test_jaxn_stripper():
    import json

    from ofiqpy.config import _strip_jaxn
    txt = '{ // comment\n "a": 1, /* block */ "b": [1, 2,], }'
    assert json.loads(_strip_jaxn(txt)) == {"a": 1, "b": [1, 2]}

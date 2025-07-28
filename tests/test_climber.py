import numpy as np
import pytest
from climber import Climber

class DummyPES:
    def __init__(self, atoms=None): pass
    def energy(self, coords): return float(np.sum(coords**2))
    def forces(self, coords): return 2*coords


def test_random_mode_normalized():
    coords = np.zeros((4,3))
    climber = Climber(DummyPES(), ds_atom=0.5, W=1.0, NG=2, Ratio_local=1.0)
    mode = climber.random_mode(coords)
    assert pytest.approx(1.0, rel=1e-6) == np.linalg.norm(mode)


def test_climb_returns_structure():
    coords = np.zeros((2,3))
    coords[1] = [1.0, 0.0, 0.0]
    climber = Climber(DummyPES(), ds_atom=0.5, W=1.0, NG=2, Ratio_local=1.0)
    final_min, intermediates, E_start = climber.climb(coords)
    assert isinstance(final_min, np.ndarray)
    assert isinstance(intermediates, list)
    assert isinstance(E_start, float)
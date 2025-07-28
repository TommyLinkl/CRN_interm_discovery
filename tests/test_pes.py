import numpy as np
import pytest
from pes import LJPotential
from ase import Atoms


def test_lj_potential_interface():
    atoms = Atoms('Ar2', positions=[(0,0,0),(3,0,0)])
    pot   = LJPotential(atoms)
    coords = atoms.get_positions()
    e     = pot.energy(coords)
    f     = pot.forces(coords)
    assert isinstance(e, float)
    assert isinstance(f, np.ndarray)
    assert f.shape == coords.shape
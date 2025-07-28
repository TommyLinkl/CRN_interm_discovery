import numpy as np
from relax import LocalRelaxer
from ase import Atoms


def test_relax_returns_same_shape():
    atoms   = Atoms('Ar', positions=[(0,0,0)])
    relaxer = LocalRelaxer(atoms, fmax=0.01, steps=1)
    coords  = np.array([[0.0, 0.0, 0.0]])
    result  = relaxer.relax(coords)
    assert isinstance(result, np.ndarray)
    assert result.shape == coords.shape

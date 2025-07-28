import numpy as np
from mc import MetropolisMC


def test_accept_downhill():
    assert MetropolisMC.accept(5.0, 3.0, 300)

def test_accept_uphill_probability(monkeypatch):
    monkeypatch.setattr('numpy.random.rand', lambda: 0.0)
    assert MetropolisMC.accept(3.0, 5.0, 300)
    monkeypatch.setattr('numpy.random.rand', lambda: 1.0)
    assert not MetropolisMC.accept(3.0, 5.0, 300)
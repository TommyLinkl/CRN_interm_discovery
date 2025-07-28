import numpy as np
from abc import ABC, abstractmethod

class Potential(ABC):
    """Abstract base class for potential energy surfaces."""
    
    @abstractmethod
    def energy(self, coords: np.ndarray) -> float:
        """Calculate energy at given coordinates.
        
        Args:
            coords: Atomic coordinates, shape (N_atoms, 3)
            
        Returns:
            Energy value
        """
        pass

    @abstractmethod
    def forces(self, coords: np.ndarray) -> np.ndarray:
        """Calculate forces at given coordinates.
        
        Args:
            coords: Atomic coordinates, shape (N_atoms, 3)
            
        Returns:
            Forces array, shape (N_atoms, 3)
        """
        pass

class LJPotential(Potential):
    """Lennard-Jones potential implementation using ASE."""
    
    def __init__(self, atoms, epsilon=1.0, sigma=2.7):
        """Initialize LJ potential.
        
        Args:
            atoms: ASE Atoms object
            epsilon: LJ energy parameter
            sigma: LJ distance parameter
        """
        from ase.calculators.lj import LennardJones
        self.atoms = atoms
        self.atoms.calc = LennardJones(epsilon=epsilon, sigma=sigma)

    def energy(self, coords: np.ndarray) -> float:
        """Calculate LJ energy."""
        self.atoms.set_positions(coords)
        return self.atoms.get_potential_energy()

    def forces(self, coords: np.ndarray) -> np.ndarray:
        """Calculate LJ forces."""
        self.atoms.set_positions(coords)
        return self.atoms.get_forces()
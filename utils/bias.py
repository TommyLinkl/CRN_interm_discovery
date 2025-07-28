import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from typing import List, Tuple

class BiasPotential:
    """Directional Gaussian bias potential for enhanced sampling."""
    
    def __init__(self, ds_atom: float, W: float, NG: int):
        """Initialize bias potential.
        
        Args:
            ds_atom: Gaussian width parameter
            W: Gaussian height parameter
            NG: Maximum number of Gaussian hills
        """
        self.ds = ds_atom
        self.W = W
        self.NG = NG
        self.hills: List[Tuple[np.ndarray, np.ndarray]] = []

    def add_gaussian(self, center: np.ndarray, direction: np.ndarray):
        """Add a directional Gaussian hill.
        
        Args:
            center: Center coordinates, shape (N_atoms, 3)
            direction: Direction vector, shape (N_atoms, 3)
        """
        if len(self.hills) < self.NG:
            direction_flat = direction.reshape(-1)
            direction_flat /= np.linalg.norm(direction_flat)
            unit_direction = direction_flat.reshape(direction.shape)
            self.hills.append((center.copy(), unit_direction))

    def clear(self):
        """Remove all Gaussian hills."""
        self.hills.clear()

    def energy(self, coords: np.ndarray) -> float:
        """Calculate bias energy at given coordinates.
        
        Args:
            coords: Coordinates, shape (N_atoms, 3)
            
        Returns:
            Bias energy
        """
        energy_total = 0.0
        for center, direction in self.hills:
            dr_flat = (coords - center).reshape(-1)
            direction_flat = direction.reshape(-1)
            projection = float(np.dot(dr_flat, direction_flat))
            energy_total += self.W * np.exp(-(projection**2) / (2 * self.ds**2))
        return energy_total

    def forces(self, coords: np.ndarray) -> np.ndarray:
        """Calculate bias forces at given coordinates.
        
        Args:
            coords: Coordinates, shape (N_atoms, 3)
            
        Returns:
            Bias forces, shape (N_atoms, 3)
        """
        forces_total = np.zeros_like(coords)
        for center, direction in self.hills:
            dr_flat = (coords - center).reshape(-1)
            direction_flat = direction.reshape(-1)
            projection = float(np.dot(dr_flat, direction_flat))
            prefactor = (
                (self.W / self.ds**2) * 
                np.exp(-projection**2 / (2 * self.ds**2)) * 
                projection
            )
            forces_total += (prefactor * direction_flat).reshape(coords.shape)
        return forces_total

class BiasCalculator(Calculator):
    """ASE Calculator wrapper that adds bias potential to base calculator."""
    
    implemented_properties = ['energy', 'forces']

    def __init__(self, base_calc, bias_potential: BiasPotential):
        """Initialize biased calculator.
        
        Args:
            base_calc: Base ASE calculator
            bias_potential: Bias potential to add
        """
        super().__init__()
        self.base_calc = base_calc
        self.bias = bias_potential

    def calculate(self, atoms, properties=None, system_changes=all_changes):
        """Calculate energy and forces with bias."""
        self.base_calc.calculate(atoms, properties, system_changes)
        
        base_energy = self.base_calc.results['energy']
        base_forces = self.base_calc.results['forces']
        
        coords = atoms.get_positions()
        bias_energy = self.bias.energy(coords)
        bias_forces = self.bias.forces(coords)
        
        self.results['energy'] = base_energy + bias_energy
        self.results['forces'] = base_forces + bias_forces
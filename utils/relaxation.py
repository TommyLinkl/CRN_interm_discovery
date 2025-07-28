import numpy as np
from ase import Atoms
from ase.optimize import BFGS

class LocalRelaxer:
    """Local geometry optimization using ASE BFGS optimizer."""
    
    def __init__(self, atoms: Atoms, fmax: float = 0.05, steps: int = 500):
        """Initialize local relaxer.
        
        Args:
            atoms: ASE Atoms object
            fmax: Force convergence criterion
            steps: Maximum optimization steps
        """
        self.atoms = atoms
        self.fmax = fmax
        self.steps = steps

    def relax(self, coords: np.ndarray) -> np.ndarray:
        """Perform local relaxation starting from given coordinates.
        
        Args:
            coords: Initial coordinates, shape (N_atoms, 3)
            
        Returns:
            Relaxed coordinates, shape (N_atoms, 3)
        """
        self.atoms.set_positions(coords)
        optimizer = BFGS(self.atoms, logfile=None)
        optimizer.run(fmax=self.fmax, steps=self.steps)
        return self.atoms.get_positions()
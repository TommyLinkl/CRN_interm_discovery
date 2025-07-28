import numpy as np
from ase.io import read, write
from ase import Atoms
from typing import List, Tuple
from core import Configuration, ConfigurationSet

def load_xyz(filepath: str) -> Tuple[np.ndarray, Atoms]:
    """Load coordinates and atoms from XYZ file.
    
    Args:
        filepath: Path to XYZ file
        
    Returns:
        Tuple of (coordinates, atoms)
    """
    atoms = read(filepath)
    coords = atoms.get_positions()
    return coords, atoms

def save_xyz(filepath: str, coords: np.ndarray, symbols: List[str]):
    """Save coordinates to XYZ file.
    
    Args:
        filepath: Output XYZ file path
        coords: Atomic coordinates, shape (N_atoms, 3)
        symbols: List of atomic symbols
    """
    atoms = Atoms(symbols=symbols, positions=coords)
    write(filepath, atoms)

class ConfigurationIO:
    """I/O utilities for Configuration objects."""
    
    @staticmethod
    def save_configurations(filepath: str, config_set: ConfigurationSet, 
                          symbols: List[str]):
        """Save configurations to multi-frame XYZ file.
        
        Args:
            filepath: Output XYZ file path
            config_set: Set of configurations to save
            symbols: List of atomic symbols
        """
        with open(filepath, 'w') as f:
            for i, config in enumerate(config_set):
                n_atoms = len(config.coords)
                f.write(f"{n_atoms}\n")
                
                comment = f"Energy: {config.energy:.6f} eV, Method: {config.method}"
                if config.step is not None:
                    comment += f", Step: {config.step}"
                if config.trajectory_id is not None:
                    comment += f", Traj: {config.trajectory_id}"
                f.write(f"{comment}\n")
                
                for sym, (x, y, z) in zip(symbols, config.coords):
                    f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")
    
    @staticmethod
    def save_trajectory(filepath: str, configurations: List[Configuration], 
                       symbols: List[str], trajectory_id: int):
        """Save single trajectory to XYZ file.
        
        Args:
            filepath: Output XYZ file path
            configurations: List of configurations in trajectory
            symbols: List of atomic symbols
            trajectory_id: Trajectory identifier
        """
        config_set = ConfigurationSet()
        for config in configurations:
            config.trajectory_id = trajectory_id
            config_set.add(config)
        
        ConfigurationIO.save_configurations(filepath, config_set, symbols)
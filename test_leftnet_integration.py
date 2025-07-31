#!/usr/bin/env python3

import numpy as np
from ase import Atoms
from core.pes import LeftNetPotential

def test_leftnet_potential():
    """Test LeftNet potential integration."""
    
    # Create a simple test system (e.g., water molecule)
    atoms = Atoms('H2O', positions=[(0.0, 0.0, 0.0), 
                                   (0.96, 0.0, 0.0), 
                                   (-0.24, 0.93, 0.0)])
    
    try:
        # Initialize LeftNet potential
        pot = LeftNetPotential(atoms)
        
        # Test coordinates
        coords = atoms.get_positions()
        
        # Test energy calculation
        energy = pot.energy(coords)
        print(f"Energy: {energy:.6f}")
        
        # Test force calculation  
        forces = pot.forces(coords)
        print(f"Forces shape: {forces.shape}")
        print(f"Forces:\n{forces}")
        
        # Verify types and shapes
        assert isinstance(energy, float), "Energy should be a float"
        assert isinstance(forces, np.ndarray), "Forces should be numpy array"
        assert forces.shape == coords.shape, "Forces shape should match coordinates"
        
        print("✓ LeftNet potential integration test passed!")
        return True
        
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Please ensure LeftNet_ase_calculator.py is in your Python path")
        return False
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        print("Please ensure the LeftNet checkpoint file exists")
        return False
    except Exception as e:
        print(f"Error during test: {e}")
        return False

if __name__ == "__main__":
    test_leftnet_potential()
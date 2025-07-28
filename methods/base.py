import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from core import Potential, Configuration, ConfigurationSet

class DiscoveryMethod(ABC):
    """Abstract base class for stable product discovery methods."""
    
    def __init__(self, potential: Potential, **kwargs):
        """Initialize discovery method.
        
        Args:
            potential: Potential energy surface
            **kwargs: Method-specific parameters
        """
        self.potential = potential
        self.parameters = kwargs
        self.results = ConfigurationSet()
    
    @abstractmethod
    def discover(self, initial_coords: np.ndarray, n_steps: int, **kwargs) -> ConfigurationSet:
        """Discover stable products from initial configuration.
        
        Args:
            initial_coords: Initial molecular coordinates, shape (N_atoms, 3)
            n_steps: Number of discovery steps
            **kwargs: Additional method-specific parameters
            
        Returns:
            Set of discovered configurations
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the discovery method."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters."""
        return self.parameters.copy()
    
    def reset_results(self):
        """Clear stored results."""
        self.results = ConfigurationSet()
    
    def _create_configuration(self, coords: np.ndarray, step: int = None, 
                            trajectory_id: int = None) -> Configuration:
        """Create a Configuration object with energy calculation."""
        energy = self.potential.energy(coords)
        return Configuration(
            coords=coords.copy(),
            energy=energy,
            method=self.get_method_name(),
            step=step,
            trajectory_id=trajectory_id
        )
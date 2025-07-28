import numpy as np
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Configuration:
    """Represents a molecular configuration with energy and metadata."""
    
    coords: np.ndarray  # Shape (N_atoms, 3)
    energy: float
    method: str  # Method used to generate this configuration
    step: Optional[int] = None
    trajectory_id: Optional[int] = None
    
    def copy(self):
        """Create a deep copy of the configuration."""
        return Configuration(
            coords=self.coords.copy(),
            energy=self.energy,
            method=self.method,
            step=self.step,
            trajectory_id=self.trajectory_id
        )

class ConfigurationSet:
    """Container for managing collections of molecular configurations."""
    
    def __init__(self):
        self.configurations: List[Configuration] = []
    
    def add(self, config: Configuration):
        """Add a configuration to the set."""
        self.configurations.append(config)
    
    def extend(self, configs: List[Configuration]):
        """Add multiple configurations to the set."""
        self.configurations.extend(configs)
    
    def get_lowest_energy(self) -> Optional[Configuration]:
        """Get the configuration with lowest energy."""
        if not self.configurations:
            return None
        return min(self.configurations, key=lambda c: c.energy)
    
    def filter_by_energy(self, max_energy: float) -> 'ConfigurationSet':
        """Filter configurations by maximum energy."""
        filtered = ConfigurationSet()
        filtered.configurations = [c for c in self.configurations if c.energy <= max_energy]
        return filtered
    
    def filter_by_method(self, method: str) -> 'ConfigurationSet':
        """Filter configurations by method."""
        filtered = ConfigurationSet()
        filtered.configurations = [c for c in self.configurations if c.method == method]
        return filtered
    
    def __len__(self) -> int:
        return len(self.configurations)
    
    def __iter__(self):
        return iter(self.configurations)
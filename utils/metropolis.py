import numpy as np

class MetropolisAcceptance:
    """Metropolis acceptance criterion for Monte Carlo moves."""
    
    @staticmethod
    def accept(energy_old: float, energy_new: float, temperature: float) -> bool:
        """Determine acceptance based on Metropolis criterion.
        
        Args:
            energy_old: Current energy
            energy_new: Proposed energy
            temperature: Temperature for acceptance probability
            
        Returns:
            True if move should be accepted, False otherwise
        """
        if energy_new < energy_old:
            return True
        
        delta_energy = energy_new - energy_old
        probability = np.exp(-delta_energy / temperature)
        return np.random.rand() < probability
    
    @staticmethod
    def acceptance_probability(energy_old: float, energy_new: float, 
                             temperature: float) -> float:
        """Calculate acceptance probability.
        
        Args:
            energy_old: Current energy
            energy_new: Proposed energy
            temperature: Temperature
            
        Returns:
            Acceptance probability [0, 1]
        """
        if energy_new < energy_old:
            return 1.0
        
        delta_energy = energy_new - energy_old
        return np.exp(-delta_energy / temperature)
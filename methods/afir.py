import numpy as np
from typing import List, Tuple, Optional
from core import Potential, Configuration, ConfigurationSet
from utils import LocalRelaxer, MetropolisAcceptance
from .base import DiscoveryMethod

class ArtificialForceInducedReaction(DiscoveryMethod):
    """Artificial Force Induced Reaction (AFIR) method for discovering stable products."""
    
    def __init__(self, potential: Potential, atoms, **kwargs):
        """Initialize AFIR method.
        
        Args:
            potential: Potential energy surface
            atoms: ASE Atoms object
            **kwargs: AFIR parameters
        """
        default_params = {
            'alpha': 0.1,          # Force scaling parameter
            'push_fragments': [],   # Indices of atoms to push apart
            'pull_fragments': [],   # Indices of atoms to pull together
            'temperature': 1000.0,  # Temperature for acceptance
            'max_force': 5.0,      # Maximum artificial force magnitude
            'fmax': 0.05,          # Relaxation force threshold
            'relax_steps': 500,    # Maximum relaxation steps
            'energy_window': 50.0  # Energy window for acceptance
        }
        default_params.update(kwargs)
        super().__init__(potential, **default_params)
        
        self.atoms = atoms
        self.relaxer = LocalRelaxer(
            atoms,
            fmax=self.parameters['fmax'],
            steps=self.parameters['relax_steps']
        )
    
    def get_method_name(self) -> str:
        return "AFIR"
    
    def discover(self, initial_coords: np.ndarray, n_steps: int, 
                trajectory_id: int = 0, **kwargs) -> ConfigurationSet:
        """Run AFIR discovery from initial configuration.
        
        Args:
            initial_coords: Initial coordinates, shape (N_atoms, 3)
            n_steps: Number of AFIR steps
            trajectory_id: Trajectory identifier
            **kwargs: Additional parameters
            
        Returns:
            Set of discovered configurations
        """
        coords = initial_coords.copy()
        energy_ref = self.potential.energy(coords)
        discovered = ConfigurationSet()
        
        discovered.add(self._create_configuration(coords, 0, trajectory_id))
        
        for step in range(n_steps):
            new_config = self._afir_step(coords, step, trajectory_id)
            
            if new_config is None:
                continue
                
            energy_new = new_config.energy
            delta_energy = energy_new - energy_ref
            
            if (delta_energy <= self.parameters['energy_window'] and
                MetropolisAcceptance.accept(energy_ref, energy_new, 
                                          self.parameters['temperature'])):
                coords = new_config.coords
                energy_ref = energy_new
                discovered.add(new_config)
        
        self.results.extend(discovered.configurations)
        return discovered
    
    def _afir_step(self, coords: np.ndarray, step: int, 
                  trajectory_id: int) -> Optional[Configuration]:
        """Perform one AFIR step with artificial forces."""
        from ase.calculators.calculator import Calculator, all_changes
        
        original_calc = self.relaxer.atoms.calc
        
        class AFIRCalculator(Calculator):
            implemented_properties = ['energy', 'forces']
            
            def __init__(self, base_calc, afir_method):
                super().__init__()
                self.base_calc = base_calc
                self.afir = afir_method
            
            def calculate(self, atoms, properties=None, system_changes=all_changes):
                self.base_calc.calculate(atoms, properties, system_changes)
                
                base_energy = self.base_calc.results['energy']
                base_forces = self.base_calc.results['forces']
                
                coords = atoms.get_positions()
                artificial_forces = self.afir._calculate_artificial_forces(coords)
                
                self.results['energy'] = base_energy
                self.results['forces'] = base_forces + artificial_forces
        
        afir_calc = AFIRCalculator(original_calc, self)
        self.relaxer.atoms.calc = afir_calc
        
        try:
            new_coords = self._apply_artificial_force_and_relax(coords)
            
            self.relaxer.atoms.calc = original_calc
            final_coords = self.relaxer.relax(new_coords)
            
            return self._create_configuration(final_coords, step + 1, trajectory_id)
            
        except Exception:
            self.relaxer.atoms.calc = original_calc
            return None
    
    def _calculate_artificial_forces(self, coords: np.ndarray) -> np.ndarray:
        """Calculate artificial forces for AFIR."""
        artificial_forces = np.zeros_like(coords)
        
        if self.parameters['push_fragments']:
            artificial_forces += self._calculate_push_forces(coords)
        
        if self.parameters['pull_fragments']:
            artificial_forces += self._calculate_pull_forces(coords)
        
        if not self.parameters['push_fragments'] and not self.parameters['pull_fragments']:
            artificial_forces = self._calculate_default_forces(coords)
        
        force_magnitude = np.linalg.norm(artificial_forces.reshape(-1))
        if force_magnitude > self.parameters['max_force']:
            artificial_forces *= self.parameters['max_force'] / force_magnitude
        
        return artificial_forces
    
    def _calculate_push_forces(self, coords: np.ndarray) -> np.ndarray:
        """Calculate forces to push fragments apart."""
        forces = np.zeros_like(coords)
        alpha = self.parameters['alpha']
        
        for fragment in self.parameters['push_fragments']:
            if len(fragment) < 2:
                continue
                
            center = coords[fragment].mean(axis=0)
            
            for i in fragment:
                direction = coords[i] - center
                distance = np.linalg.norm(direction)
                if distance > 1e-6:
                    direction_normalized = direction / distance
                    force_magnitude = alpha / (distance + 1e-6)
                    forces[i] += force_magnitude * direction_normalized
        
        return forces
    
    def _calculate_pull_forces(self, coords: np.ndarray) -> np.ndarray:
        """Calculate forces to pull fragments together."""
        forces = np.zeros_like(coords)
        alpha = self.parameters['alpha']
        
        if len(self.parameters['pull_fragments']) < 2:
            return forces
        
        centers = []
        for fragment in self.parameters['pull_fragments']:
            if len(fragment) > 0:
                centers.append(coords[fragment].mean(axis=0))
        
        if len(centers) < 2:
            return forces
        
        for i, fragment in enumerate(self.parameters['pull_fragments']):
            if len(fragment) == 0:
                continue
                
            target_center = np.mean([c for j, c in enumerate(centers) if j != i], axis=0)
            
            for atom_idx in fragment:
                direction = target_center - coords[atom_idx]
                distance = np.linalg.norm(direction)
                if distance > 1e-6:
                    direction_normalized = direction / distance
                    force_magnitude = alpha * distance
                    forces[atom_idx] += force_magnitude * direction_normalized
        
        return forces
    
    def _calculate_default_forces(self, coords: np.ndarray) -> np.ndarray:
        """Calculate default artificial forces when no fragments specified."""
        forces = np.zeros_like(coords)
        alpha = self.parameters['alpha']
        
        n_atoms = len(coords)
        if n_atoms < 2:
            return forces
        
        center_of_mass = coords.mean(axis=0)
        
        for i in range(n_atoms):
            direction = coords[i] - center_of_mass
            distance = np.linalg.norm(direction)
            if distance > 1e-6:
                direction_normalized = direction / distance
                force_magnitude = alpha * np.random.uniform(0.5, 1.5)
                forces[i] += force_magnitude * direction_normalized
        
        return forces
    
    def _apply_artificial_force_and_relax(self, coords: np.ndarray) -> np.ndarray:
        """Apply artificial forces and perform constrained relaxation."""
        displacement_scale = 0.01
        artificial_forces = self._calculate_artificial_forces(coords)
        
        displaced_coords = coords + displacement_scale * artificial_forces
        
        relaxed_coords = self.relaxer.relax(displaced_coords)
        
        return relaxed_coords
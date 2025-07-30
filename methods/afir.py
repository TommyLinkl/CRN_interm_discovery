import numpy as np
import time
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
            'random_pull_pairs': True,  # Randomly select atom pairs to pull
            'pull_pair_distance_min': 2.0,  # Minimum distance for pull pairs (Å)
            'temperature': 1000.0,  # Temperature for acceptance
            'max_force': 5.0,      # Maximum artificial force magnitude
            'fmax': 0.1,           # Faster relaxation
            'relax_steps': 100,    # Halved for speed
            'fmax_constrained': 0.2,  # Even faster for artificial force relaxation
            'relax_steps_constrained': 50,   # Halved for maximum speed
            'energy_window': 50.0  # Energy window for acceptance
        }
        default_params.update(kwargs)
        super().__init__(potential, **default_params)
        
        self.atoms = atoms
        # Final relaxer (more accurate)
        self.relaxer = LocalRelaxer(
            atoms,
            fmax=self.parameters['fmax'],
            steps=self.parameters['relax_steps']
        )
        
        # Constrained relaxer (faster for artificial force steps)
        self.constrained_relaxer = LocalRelaxer(
            atoms,
            fmax=self.parameters['fmax_constrained'],
            steps=self.parameters['relax_steps_constrained']
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
        
        total_time = 0.0
        force_calc_time = 0.0
        relax_time = 0.0
        
        for step in range(n_steps):
            new_config = self._afir_step(coords, step, trajectory_id)
            
            if new_config[0] is None:
                continue
                
            config, step_timings = new_config
            if step_timings:
                force_calc_time += step_timings['force_time']
                relax_time += step_timings['relax_time']
            
            energy_new = config.energy
            delta_energy = energy_new - energy_ref
            
            if (delta_energy <= self.parameters['energy_window'] and
                MetropolisAcceptance.accept(energy_ref, energy_new, 
                                          self.parameters['temperature'])):
                coords = config.coords
                energy_ref = energy_new
                discovered.add(config)
        
        self.results.extend(discovered.configurations)
        
        total_time = force_calc_time + relax_time
        
        if n_steps > 0 and total_time > 0:
            print(f"AFIR Timing Summary (n_steps={n_steps}):")
            print(f"  Total time: {total_time:.3f}s ({total_time/n_steps:.3f}s/step)")
            print(f"  Force calculation: {force_calc_time:.3f}s ({force_calc_time/total_time*100:.1f}%)")
            print(f"  Relaxation: {relax_time:.3f}s ({relax_time/total_time*100:.1f}%)")
        
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
            # Time artificial force application and relaxation
            force_start = time.time()
            new_coords = self._apply_artificial_force_and_relax(coords)
            relax_with_forces_time = time.time() - force_start
            
            # Time final relaxation
            final_relax_start = time.time()
            self.constrained_relaxer.atoms.calc = original_calc
            final_coords = self.relaxer.relax(new_coords)
            final_relax_time = time.time() - final_relax_start
            
            timings = {
                'force_time': relax_with_forces_time,
                'relax_time': final_relax_time
            }
            
            return self._create_configuration(final_coords, step + 1, trajectory_id), timings
            
        except Exception:
            self.constrained_relaxer.atoms.calc = original_calc
            return None, None
    
    def _calculate_artificial_forces(self, coords: np.ndarray) -> np.ndarray:
        """Calculate artificial forces for AFIR."""
        artificial_forces = np.zeros_like(coords)
        
        if self.parameters['push_fragments']:
            artificial_forces += self._calculate_push_forces(coords)
        
        if self.parameters['pull_fragments']:
            artificial_forces += self._calculate_pull_forces(coords)
        
        # When no specific fragments are defined, use default radial forces
        if not self.parameters['push_fragments'] and not self.parameters['pull_fragments']:
            artificial_forces = self._calculate_default_forces(coords)
            
            # Add random pull forces on top of default radial forces
            if self.parameters['random_pull_pairs']:
                artificial_forces += self._calculate_random_pull_forces(coords)
        
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
        
        # Use faster relaxer for artificial force relaxation
        relaxed_coords = self.constrained_relaxer.relax(displaced_coords)
        
        return relaxed_coords
    
    def _calculate_random_pull_forces(self, coords: np.ndarray) -> np.ndarray:
        """Calculate additional pull forces between randomly selected atom pairs."""
        forces = np.zeros_like(coords)
        alpha = self.parameters['alpha']
        min_distance = self.parameters['pull_pair_distance_min']
        n_atoms = len(coords)
        
        if n_atoms < 2:
            return forces
        
        # Find suitable atom pairs (separated by at least min_distance)
        suitable_pairs = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance >= min_distance:
                    suitable_pairs.append((i, j, distance))
        
        if not suitable_pairs:
            return forces  # No suitable pairs found
        
        # Randomly select one pair to pull together
        atom_i, atom_j, current_distance = suitable_pairs[np.random.randint(len(suitable_pairs))]
        
        # Calculate pull force (attractive, proportional to distance)
        direction = coords[atom_j] - coords[atom_i]
        direction_normalized = direction / current_distance
        
        # Force magnitude increases with distance (encourages bond formation)
        force_magnitude = alpha * current_distance * 0.5  # Scale down to avoid overwhelming radial forces
        
        # Apply equal and opposite forces
        forces[atom_i] += force_magnitude * direction_normalized
        forces[atom_j] -= force_magnitude * direction_normalized
        
        return forces
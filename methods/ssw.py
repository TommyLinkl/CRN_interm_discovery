import numpy as np
import time
from typing import List
from ase.units import kB
from core import Potential, Configuration, ConfigurationSet
from utils import LocalRelaxer, BiasPotential, BiasCalculator, MetropolisAcceptance
from .base import DiscoveryMethod

class StochasticSurfaceWalk(DiscoveryMethod):
    """Stochastic Surface Walk method for discovering stable products."""
    
    def __init__(self, potential: Potential, atoms, **kwargs):
        """Initialize SSW method.
        
        Args:
            potential: Potential energy surface
            atoms: ASE Atoms object
            **kwargs: SSW parameters
        """
        default_params = {
            'ds_atom': 0.5,
            'W': 1.0,
            'NG': 8,
            'temperature': 3000.0,
            'energy_window': 20.0,
            'ratio_local': 100.0,
            'fmax': 0.05,
            'relax_steps': 500,
            'delta_R': 0.005,
            'rot_tol': 1e-3,
            'rot_max_iter': 10
        }
        default_params.update(kwargs)
        super().__init__(potential, **default_params)
        
        self.atoms = atoms
        self.bias = BiasPotential(
            self.parameters['ds_atom'],
            self.parameters['W'],
            self.parameters['NG']
        )
        self.relaxer = LocalRelaxer(
            atoms,
            fmax=self.parameters['fmax'],
            steps=self.parameters['relax_steps']
        )
    
    def get_method_name(self) -> str:
        return "SSW"
    
    def discover(self, initial_coords: np.ndarray, n_steps: int, 
                trajectory_id: int = 0, **kwargs) -> ConfigurationSet:
        """Run SSW discovery from initial configuration.
        
        Args:
            initial_coords: Initial coordinates, shape (N_atoms, 3)
            n_steps: Number of SSW steps
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
        mode_time = 0.0
        relax_time = 0.0
        
        for step in range(n_steps):
            step_start = time.time()
            new_min, step_timings = self._climb_step(coords)
            step_time = time.time() - step_start
            
            total_time += step_time
            mode_time += step_timings['mode_time']
            relax_time += step_timings['relax_time']
            energy_new = self.potential.energy(new_min)
            delta_energy = energy_new - energy_ref
            
            if delta_energy > self.parameters['energy_window']:
                continue
                
            if MetropolisAcceptance.accept(energy_ref, energy_new, 
                                          self.parameters['temperature']):
                coords = new_min
                energy_ref = energy_new
                
                # Only save the final minimum from this MC step
                final_config = self._create_configuration(new_min, step + 1, trajectory_id)
                discovered.add(final_config)
        
        self.results.extend(discovered.configurations)
        
        if n_steps > 0:
            print(f"SSW Timing Summary (n_steps={n_steps}):")
            print(f"  Total time: {total_time:.3f}s ({total_time/n_steps:.3f}s/step)")
            print(f"  Mode generation: {mode_time:.3f}s ({mode_time/total_time*100:.1f}%)")
            print(f"  Relaxation: {relax_time:.3f}s ({relax_time/total_time*100:.1f}%)")
        
        return discovered
    
    def _climb_step(self, coords: np.ndarray):
        """Perform one SSW climbing step."""
        current = coords.copy()
        energy_start = self.potential.energy(current)
        
        # Time mode generation
        mode_start = time.time()
        mode = self._generate_random_mode(current)
        mode_time = time.time() - mode_start
        
        original_calc = self.relaxer.atoms.calc
        self.bias.clear()
        
        relax_time = 0.0
        
        for _ in range(self.parameters['NG']):
            mode = self._refine_mode(current, mode)
            self.bias.add_gaussian(center=current, direction=mode)
            trial = current + mode * self.parameters['ds_atom']
            
            # Time biased relaxation
            relax_start = time.time()
            biased_calc = BiasCalculator(original_calc, self.bias)
            self.relaxer.atoms.calc = biased_calc
            trial = self.relaxer.relax(trial)
            relax_time += time.time() - relax_start
            
            current = trial
        
        # Time final relaxation
        final_relax_start = time.time()
        self.relaxer.atoms.calc = original_calc
        final_min = self.relaxer.relax(current)
        relax_time += time.time() - final_relax_start
        
        timings = {
            'mode_time': mode_time,
            'relax_time': relax_time
        }
        
        return final_min, timings
    
    def _generate_random_mode(self, coords: np.ndarray) -> np.ndarray:
        """Generate random displacement mode combining global and local."""
        global_mode = self._sample_global_mode()
        local_mode = self._sample_local_mode(coords)
        
        mode = global_mode + self.parameters['ratio_local'] * local_mode
        mode_flat = mode.reshape(-1)
        mode_flat /= np.linalg.norm(mode_flat)
        return mode_flat.reshape(coords.shape)
    
    def _sample_global_mode(self, temperature: float = 300.0) -> np.ndarray:
        """Sample global vibrational mode."""
        coords = self.atoms.get_positions()
        masses = self.atoms.get_masses()
        
        sigma = np.sqrt(kB * temperature / masses)[:, None]
        velocities = np.random.normal(size=coords.shape) * sigma
        
        total_mass = masses.sum()
        v_com = (masses[:, None] * velocities).sum(axis=0) / total_mass
        velocities -= v_com
        
        kinetic_actual = 0.5 * (masses[:, None] * velocities**2).sum()
        kinetic_target = 0.5 * (3 * len(masses) - 3) * kB * temperature
        
        if kinetic_actual > 0:
            velocities *= np.sqrt(kinetic_target / kinetic_actual)
        
        velocities_flat = velocities.reshape(-1)
        velocities_flat /= np.linalg.norm(velocities_flat)
        return velocities_flat.reshape(coords.shape)
    
    def _sample_local_mode(self, coords: np.ndarray, max_tries: int = 100, 
                          min_dist: float = 3.0) -> np.ndarray:
        """Sample local pairwise mode."""
        for _ in range(max_tries):
            i, j = np.random.choice(len(coords), 2, replace=False)
            if np.linalg.norm(coords[i] - coords[j]) > min_dist:
                delta = coords[j] - coords[i]
                mode = np.zeros_like(coords)
                mode[i] = delta
                mode[j] = -delta
                return mode
        
        return np.zeros_like(coords)
    
    def _refine_mode(self, coords: np.ndarray, mode: np.ndarray) -> np.ndarray:
        """Refine mode using dimer rotation."""
        current_mode = mode.copy()
        mode_flat = current_mode.reshape(-1)
        
        for _ in range(self.parameters['rot_max_iter']):
            R1 = coords + self.parameters['delta_R'] * current_mode
            R2 = coords - self.parameters['delta_R'] * current_mode
            
            F1 = self.potential.forces(R1)
            F2 = self.potential.forces(R2)
            
            dF = (F2 - F1).reshape(-1)
            projection = np.dot(dF, mode_flat)
            dF_perp = dF - projection * mode_flat
            
            if np.linalg.norm(dF_perp) < self.parameters['rot_tol']:
                break
            
            gamma = 1.0 / (2 * self.parameters['delta_R'])
            mode_flat = mode_flat + gamma * dF_perp
            mode_flat /= np.linalg.norm(mode_flat)
            current_mode = mode_flat.reshape(mode.shape)
        
        return current_mode
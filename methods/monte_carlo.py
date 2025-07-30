import numpy as np
import time
from typing import List, Optional
from core import Potential, Configuration, ConfigurationSet
from utils import LocalRelaxer, MetropolisAcceptance
from .base import DiscoveryMethod

class MonteCarloDiscovery(DiscoveryMethod):
    """Plain Monte Carlo method for discovering stable products."""
    
    def __init__(self, potential: Potential, atoms, **kwargs):
        """Initialize Monte Carlo discovery method.
        
        Args:
            potential: Potential energy surface
            atoms: ASE Atoms object
            **kwargs: Monte Carlo parameters
        """
        default_params = {
            'temperature': 1000.0,     # Temperature for MC moves
            'step_size': 0.1,          # Maximum displacement per move
            'relax_frequency': 10,     # Relax every N steps
            'energy_window': 50.0,     # Energy window for acceptance
            'fmax': 0.05,             # Relaxation force threshold
            'relax_steps': 500,       # Maximum relaxation steps
            'adaptive_step': True,    # Adaptive step size
            'target_acceptance': 0.5  # Target acceptance ratio
        }
        default_params.update(kwargs)
        super().__init__(potential, **default_params)
        
        self.atoms = atoms
        self.relaxer = LocalRelaxer(
            atoms,
            fmax=self.parameters['fmax'],
            steps=self.parameters['relax_steps']
        )
        
        self.acceptance_history = []
        self.current_step_size = self.parameters['step_size']
    
    def get_method_name(self) -> str:
        return "MC"
    
    def discover(self, initial_coords: np.ndarray, n_steps: int, 
                trajectory_id: int = 0, **kwargs) -> ConfigurationSet:
        """Run Monte Carlo discovery from initial configuration.
        
        Args:
            initial_coords: Initial coordinates, shape (N_atoms, 3)
            n_steps: Number of MC steps
            trajectory_id: Trajectory identifier
            **kwargs: Additional parameters
            
        Returns:
            Set of discovered configurations
        """
        coords = initial_coords.copy()
        energy_current = self.potential.energy(coords)
        discovered = ConfigurationSet()
        
        discovered.add(self._create_configuration(coords, 0, trajectory_id))
        
        n_accepted = 0
        total_time = 0.0
        move_gen_time = 0.0
        energy_eval_time = 0.0
        relax_time = 0.0
        
        for step in range(n_steps):
            # Time trial move generation
            move_start = time.time()
            trial_coords = self._generate_trial_move(coords)
            move_gen_time += time.time() - move_start
            
            # Time energy evaluation
            energy_start = time.time()
            energy_trial = self.potential.energy(trial_coords)
            energy_eval_time += time.time() - energy_start
            
            delta_energy = energy_trial - energy_current
            
            if (delta_energy <= self.parameters['energy_window'] and
                MetropolisAcceptance.accept(energy_current, energy_trial, 
                                          self.parameters['temperature'])):
                coords = trial_coords
                energy_current = energy_trial
                n_accepted += 1
                
                config = self._create_configuration(coords, step + 1, trajectory_id)
                discovered.add(config)
                
                if ((step + 1) % self.parameters['relax_frequency'] == 0):
                    relaxed_coords = self.relaxer.relax(coords)
                    relaxed_energy = self.potential.energy(relaxed_coords)
                    
                    if relaxed_energy < energy_current:
                        coords = relaxed_coords
                        energy_current = relaxed_energy
                        
                        relaxed_config = self._create_configuration(
                            coords, step + 1, trajectory_id
                        )
                        discovered.add(relaxed_config)
            
            if self.parameters['adaptive_step'] and (step + 1) % 100 == 0:
                self._adapt_step_size(n_accepted / 100)
                n_accepted = 0
        
        self.results.extend(discovered.configurations)
        return discovered
    
    def _generate_trial_move(self, coords: np.ndarray) -> np.ndarray:
        """Generate trial move using random displacement."""
        trial_coords = coords.copy()
        
        n_atoms = len(coords)
        atom_to_move = np.random.randint(n_atoms)
        
        displacement = np.random.normal(
            0, self.current_step_size, size=3
        )
        trial_coords[atom_to_move] += displacement
        
        return trial_coords
    
    def _adapt_step_size(self, acceptance_ratio: float):
        """Adapt step size based on acceptance ratio."""
        target = self.parameters['target_acceptance']
        
        if acceptance_ratio > target * 1.2:
            self.current_step_size *= 1.1
        elif acceptance_ratio < target * 0.8:
            self.current_step_size *= 0.9
        
        max_step = self.parameters['step_size'] * 5.0
        min_step = self.parameters['step_size'] * 0.1
        self.current_step_size = np.clip(
            self.current_step_size, min_step, max_step
        )
        
        self.acceptance_history.append(acceptance_ratio)

class BasinHoppingMC(MonteCarloDiscovery):
    """Basin-hopping Monte Carlo variant with periodic relaxation."""
    
    def __init__(self, potential: Potential, atoms, **kwargs):
        """Initialize basin-hopping MC method."""
        default_params = {
            'hop_size': 0.5,          # Size of basin-hopping moves
            'relax_frequency': 1,     # Relax after every move
            'temperature': 1000.0
        }
        default_params.update(kwargs)
        super().__init__(potential, atoms, **default_params)
    
    def get_method_name(self) -> str:
        return "BasinHoppingMC"
    
    def _generate_trial_move(self, coords: np.ndarray) -> np.ndarray:
        """Generate larger trial moves for basin hopping."""
        trial_coords = coords.copy()
        
        displacement = np.random.normal(
            0, self.parameters['hop_size'], size=coords.shape
        )
        
        trial_coords += displacement
        
        return self.relaxer.relax(trial_coords)

class SimulatedAnnealing(MonteCarloDiscovery):
    """Simulated annealing variant with temperature schedule."""
    
    def __init__(self, potential: Potential, atoms, **kwargs):
        """Initialize simulated annealing method."""
        default_params = {
            'initial_temperature': 2000.0,
            'final_temperature': 100.0,
            'cooling_rate': 0.95,
            'cooling_frequency': 100
        }
        default_params.update(kwargs)
        super().__init__(potential, atoms, **default_params)
        
        self.current_temperature = self.parameters['initial_temperature']
    
    def get_method_name(self) -> str:
        return "SimulatedAnnealing"
    
    def discover(self, initial_coords: np.ndarray, n_steps: int, 
                trajectory_id: int = 0, **kwargs) -> ConfigurationSet:
        """Run simulated annealing discovery."""
        self.current_temperature = self.parameters['initial_temperature']
        
        coords = initial_coords.copy()
        energy_current = self.potential.energy(coords)
        discovered = ConfigurationSet()
        
        discovered.add(self._create_configuration(coords, 0, trajectory_id))
        
        total_time = 0.0
        move_gen_time = 0.0
        energy_eval_time = 0.0
        n_accepted = 0
        
        for step in range(n_steps):
            if (step % self.parameters['cooling_frequency'] == 0 and 
                step > 0):
                self.current_temperature *= self.parameters['cooling_rate']
                self.current_temperature = max(
                    self.current_temperature,
                    self.parameters['final_temperature']
                )
            
            # Time trial move generation
            move_start = time.time()
            trial_coords = self._generate_trial_move(coords)
            move_gen_time += time.time() - move_start
            
            # Time energy evaluation
            energy_start = time.time()
            energy_trial = self.potential.energy(trial_coords)
            energy_eval_time += time.time() - energy_start
            
            if MetropolisAcceptance.accept(energy_current, energy_trial, 
                                          self.current_temperature):
                coords = trial_coords
                energy_current = energy_trial
                n_accepted += 1
                
                config = self._create_configuration(coords, step + 1, trajectory_id)
                discovered.add(config)
        
        self.results.extend(discovered.configurations)
        
        total_time = move_gen_time + energy_eval_time
        
        if n_steps > 0:
            print(f"Simulated Annealing Timing Summary (n_steps={n_steps}):")
            print(f"  Total time: {total_time:.3f}s ({total_time/n_steps:.3f}s/step)")
            print(f"  Move generation: {move_gen_time:.3f}s ({move_gen_time/total_time*100:.1f}%)")
            print(f"  Energy evaluation: {energy_eval_time:.3f}s ({energy_eval_time/total_time*100:.1f}%)")
            print(f"  Acceptance rate: {n_accepted/n_steps*100:.1f}%")
            print(f"  Final temperature: {self.current_temperature:.1f}K")
        
        return discovered
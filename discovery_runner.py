import numpy as np
import argparse
import json
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path

from core import LJPotential, ConfigurationSet
from methods import DiscoveryMethod, StochasticSurfaceWalk, ArtificialForceInducedReaction, MonteCarloDiscovery
from utils import load_xyz, ConfigurationIO

class DiscoveryRunner:
    """Main orchestrator for stable product discovery."""
    
    def __init__(self):
        self.available_methods = {
            'ssw': StochasticSurfaceWalk,
            'afir': ArtificialForceInducedReaction,
            'mc': MonteCarloDiscovery
        }
        self.results = {}
    
    def run_discovery(self, config_file: str = None, **override_params):
        """Run discovery with specified configuration.
        
        Args:
            config_file: Path to configuration file (YAML/JSON)
            **override_params: Parameters to override from config
        """
        config = self._load_config(config_file) if config_file else {}
        config.update(override_params)
        
        coords, atoms = load_xyz(config['input_xyz'])
        symbols = atoms.get_chemical_symbols()
        
        potential = LJPotential(
            atoms,
            epsilon=config.get('epsilon', 1.0),
            sigma=config.get('sigma', 2.7)
        )
        
        all_results = ConfigurationSet()
        
        for method_name in config.get('methods', ['ssw']):
            if method_name not in self.available_methods:
                print(f"Warning: Unknown method '{method_name}', skipping")
                continue
            
            method_class = self.available_methods[method_name]
            method_params = config.get(f'{method_name}_params', {})
            
            method = method_class(potential, atoms, **method_params)
            
            print(f"Running {method_name.upper()} discovery...")
            
            n_trajectories = config.get('n_trajectories', 1)
            n_steps = config.get('n_steps_per_trajectory', 1000)
            
            for traj_id in range(n_trajectories):
                traj_results = method.discover(
                    coords, n_steps, trajectory_id=traj_id
                )
                
                all_results.extend(traj_results.configurations)
                
                if config.get('save_trajectories', False):
                    traj_file = f"{config.get('output_prefix', 'discovery')}_{method_name}_traj{traj_id}.xyz"
                    ConfigurationIO.save_trajectory(
                        traj_file, traj_results.configurations, symbols, traj_id
                    )
                
                print(f"  Trajectory {traj_id}: {len(traj_results)} configurations")
            
            self.results[method_name] = method.results
        
        if config.get('save_all_results', True):
            output_file = f"{config.get('output_prefix', 'discovery')}_all_results.xyz"
            ConfigurationIO.save_configurations(output_file, all_results, symbols)
            print(f"Saved all results to {output_file}")
        
        self._print_summary(all_results)
        
        return all_results
    
    def run_method(self, method_name: str, input_xyz: str, n_steps: int = 1000,
                   n_trajectories: int = 1, output_prefix: str = "discovery",
                   **method_params) -> ConfigurationSet:
        """Run a specific discovery method.
        
        Args:
            method_name: Name of method ('ssw', 'afir', 'mc')
            input_xyz: Path to input XYZ file
            n_steps: Number of steps per trajectory
            n_trajectories: Number of trajectories
            output_prefix: Output file prefix
            **method_params: Method-specific parameters
            
        Returns:
            Set of discovered configurations
        """
        if method_name not in self.available_methods:
            raise ValueError(f"Unknown method: {method_name}")
        
        coords, atoms = load_xyz(input_xyz)
        symbols = atoms.get_chemical_symbols()
        
        potential = LJPotential(atoms)
        method_class = self.available_methods[method_name]
        method = method_class(potential, atoms, **method_params)
        
        all_results = ConfigurationSet()
        
        for traj_id in range(n_trajectories):
            traj_results = method.discover(coords, n_steps, trajectory_id=traj_id)
            all_results.extend(traj_results.configurations)
            
            traj_file = f"{output_prefix}_{method_name}_traj{traj_id}.xyz"
            ConfigurationIO.save_trajectory(
                traj_file, traj_results.configurations, symbols, traj_id
            )
        
        output_file = f"{output_prefix}_{method_name}_all.xyz"
        ConfigurationIO.save_configurations(output_file, all_results, symbols)
        
        return all_results
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError("Config file must be .yaml, .yml, or .json")
    
    def _print_summary(self, results: ConfigurationSet):
        """Print discovery summary."""
        if len(results) == 0:
            print("No configurations discovered.")
            return
        
        lowest_energy = results.get_lowest_energy()
        energies = [config.energy for config in results]
        
        print(f"\nDiscovery Summary:")
        print(f"  Total configurations: {len(results)}")
        print(f"  Energy range: {min(energies):.6f} to {max(energies):.6f} eV")
        print(f"  Lowest energy: {lowest_energy.energy:.6f} eV ({lowest_energy.method})")
        
        method_counts = {}
        for config in results:
            method_counts[config.method] = method_counts.get(config.method, 0) + 1
        
        print("  Configurations by method:")
        for method, count in method_counts.items():
            print(f"    {method}: {count}")

def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Discover stable molecular products using multiple methods"
    )
    
    parser.add_argument(
        'input_xyz',
        help="Input XYZ file with initial molecular structure"
    )
    
    parser.add_argument(
        '--config', '-c',
        help="Configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        '--methods', '-m',
        nargs='+',
        choices=['ssw', 'afir', 'mc'],
        default=['ssw'],
        help="Discovery methods to use"
    )
    
    parser.add_argument(
        '--n-trajectories', '-n',
        type=int,
        default=1,
        help="Number of trajectories per method"
    )
    
    parser.add_argument(
        '--n-steps',
        type=int,
        default=1000,
        help="Number of steps per trajectory"
    )
    
    parser.add_argument(
        '--output-prefix', '-o',
        default='discovery',
        help="Output file prefix"
    )
    
    parser.add_argument(
        '--temperature', '-T',
        type=float,
        default=1000.0,
        help="Temperature for acceptance criteria"
    )
    
    return parser

def main():
    """Main entry point for command-line usage."""
    parser = create_parser()
    args = parser.parse_args()
    
    runner = DiscoveryRunner()
    
    config_params = {
        'input_xyz': args.input_xyz,
        'methods': args.methods,
        'n_trajectories': args.n_trajectories,
        'n_steps_per_trajectory': args.n_steps,
        'output_prefix': args.output_prefix,
        'save_all_results': True,
        'save_trajectories': True
    }
    
    method_params = {
        'temperature': args.temperature
    }
    
    for method in args.methods:
        config_params[f'{method}_params'] = method_params
    
    runner.run_discovery(args.config, **config_params)

if __name__ == '__main__':
    main()
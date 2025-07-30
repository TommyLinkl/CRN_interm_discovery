#!/usr/bin/env python3
"""
Example script to run discovery using CALCS/calc_1 inputs.

This demonstrates the modular framework with a real molecular system.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core import LJPotential, ConfigurationSet
from methods import StochasticSurfaceWalk, ArtificialForceInducedReaction, MonteCarloDiscovery
from utils import load_xyz, ConfigurationIO
from discovery_runner import DiscoveryRunner

def run_individual_methods():
    """Run each method individually with the calc_1 inputs."""
    
    print("="*60)
    print("INDIVIDUAL METHOD TESTING")
    print("="*60)
    
    # Load the molecular structure
    xyz_file = "CALCS/calc_1/input.xyz"
    coords, atoms = load_xyz(xyz_file)
    symbols = atoms.get_chemical_symbols()
    
    print(f"Loaded structure: {len(atoms)} atoms")
    print(f"Atom types: {', '.join(set(symbols))}")
    
    # Set up Lennard-Jones potential
    potential = LJPotential(atoms, epsilon=1.0, sigma=2.7)
    initial_energy = potential.energy(coords)
    print(f"Initial energy: {initial_energy:.6f} eV")
    
    # Test each method with reduced steps for demonstration
    n_steps = 50
    
    # 1. Stochastic Surface Walk
    print(f"\n1. Running SSW ({n_steps} steps)...")
    ssw = StochasticSurfaceWalk(
        potential, atoms,
        ds_atom=0.5,
        W=1.0,
        NG=5,  # Reduced for faster testing
        temperature=3000.0,
        energy_window=20.0,
        ratio_local=100.0
    )
    
    ssw_results = ssw.discover(coords, n_steps=n_steps, trajectory_id=0)
    print(f"SSW found {len(ssw_results)} configurations")
    
    if len(ssw_results) > 0:
        lowest_ssw = ssw_results.get_lowest_energy()
        print(f"Lowest SSW energy: {lowest_ssw.energy:.6f} eV")
        print(f"Energy improvement: {initial_energy - lowest_ssw.energy:.6f} eV")
    
    # Save SSW results
    ConfigurationIO.save_configurations(
        "CALCS/calc_1/ssw_results.xyz", ssw_results, symbols
    )
    
    # 2. AFIR method
    print(f"\n2. Running AFIR ({n_steps} steps)...")
    afir = ArtificialForceInducedReaction(
        potential, atoms,
        alpha=0.1,
        temperature=1000.0,
        energy_window=30.0
    )
    
    afir_results = afir.discover(coords, n_steps=n_steps, trajectory_id=1)
    print(f"AFIR found {len(afir_results)} configurations")
    
    if len(afir_results) > 0:
        lowest_afir = afir_results.get_lowest_energy()
        print(f"Lowest AFIR energy: {lowest_afir.energy:.6f} eV")
        print(f"Energy improvement: {initial_energy - lowest_afir.energy:.6f} eV")
    
    # Save AFIR results
    ConfigurationIO.save_configurations(
        "CALCS/calc_1/afir_results.xyz", afir_results, symbols
    )
    
    # 3. Monte Carlo
    print(f"\n3. Running Monte Carlo ({n_steps*2} steps)...")
    mc = MonteCarloDiscovery(
        potential, atoms,
        temperature=1000.0,
        step_size=0.1,
        relax_frequency=10
    )
    
    mc_results = mc.discover(coords, n_steps=n_steps*2, trajectory_id=2)
    print(f"MC found {len(mc_results)} configurations")
    
    if len(mc_results) > 0:
        lowest_mc = mc_results.get_lowest_energy()
        print(f"Lowest MC energy: {lowest_mc.energy:.6f} eV")
        print(f"Energy improvement: {initial_energy - lowest_mc.energy:.6f} eV")
    
    # Save MC results
    ConfigurationIO.save_configurations(
        "CALCS/calc_1/mc_results.xyz", mc_results, symbols
    )
    
    # Combine all results
    all_results = ConfigurationSet()
    all_results.extend(ssw_results.configurations)
    all_results.extend(afir_results.configurations)
    all_results.extend(mc_results.configurations)
    
    if len(all_results) > 0:
        print(f"\n" + "="*60)
        print("COMBINED RESULTS SUMMARY")
        print("="*60)
        
        lowest_overall = all_results.get_lowest_energy()
        energies = [config.energy for config in all_results]
        
        print(f"Total configurations: {len(all_results)}")
        print(f"Energy range: {min(energies):.6f} to {max(energies):.6f} eV")
        print(f"Overall lowest energy: {lowest_overall.energy:.6f} eV ({lowest_overall.method})")
        print(f"Best improvement: {initial_energy - lowest_overall.energy:.6f} eV")
        
        # Count by method
        method_counts = {}
        for config in all_results:
            method_counts[config.method] = method_counts.get(config.method, 0) + 1
        
        print("\nConfigurations by method:")
        for method, count in method_counts.items():
            print(f"  {method}: {count}")
        
        # Save all results
        ConfigurationIO.save_configurations(
            "CALCS/calc_1/all_results.xyz", all_results, symbols
        )
        print(f"\nAll results saved to: CALCS/calc_1/all_results.xyz")

def run_with_config_file():
    """Run discovery using the configuration file."""
    
    print(f"\n" + "="*60)
    print("RUNNING WITH CONFIG FILE")
    print("="*60)
    
    runner = DiscoveryRunner()
    
    try:
        results = runner.run_discovery(config_file="CALCS/calc_1/config.yml")
        print(f"\nConfig-based run completed successfully!")
        print(f"Total configurations discovered: {len(results)}")
        
    except Exception as e:
        print(f"Config-based run failed: {e}")
        print("This might be due to path issues or missing dependencies.")

def main():
    """Main execution function."""
    
    print("Molecular Discovery Framework - CALCS/calc_1 Example")
    print("="*60)
    
    # Check if input file exists
    if not Path("CALCS/calc_1/input.xyz").exists():
        print("Error: CALCS/calc_1/input.xyz not found!")
        return
    
    # Create output directory if it doesn't exist
    Path("CALCS/calc_1").mkdir(parents=True, exist_ok=True)
    
    # Run individual methods first
    # run_individual_methods()
    
    # Then try config file approach
    run_with_config_file()
    
    print(f"\n" + "="*60)
    print("EXAMPLE COMPLETED")
    print("="*60)
    print("Output files created in CALCS/calc_1/:")
    print("  - discovery_results_*.xyz (from config run)")

if __name__ == '__main__':
    main()
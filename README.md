# Molecular Discovery Framework

A modular Python framework for discovering stable molecular products using computational chemistry methods.

## Overview

This framework implements three discovery methods for finding stable molecular configurations:

- **Stochastic Surface Walk (SSW)** - Enhanced sampling with directional bias potentials
- **Artificial Force Induced Reaction (AFIR)** - Chemical reaction discovery using artificial forces  
- **Monte Carlo (MC)** - Statistical sampling with Metropolis acceptance

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from core import LJPotential
from methods import StochasticSurfaceWalk
from utils import load_xyz

# Load molecular structure
coords, atoms = load_xyz("input.xyz")

# Set up potential and method
potential = LJPotential(atoms)
ssw = StochasticSurfaceWalk(potential, atoms, temperature=2000.0)

# Run discovery
results = ssw.discover(coords, n_steps=1000)
print(f"Discovered {len(results)} configurations")
```

### Command Line Usage

```bash
python discovery_runner.py input.xyz \
    --methods ssw afir mc \
    --n-trajectories 5 \
    --n-steps 1000 \
    --output-prefix discovery
```

### Configuration File

```yaml
# config.yaml
input_xyz: "input.xyz"
methods: ["ssw", "afir", "mc"]
n_trajectories: 5
n_steps_per_trajectory: 1000

ssw_params:
  temperature: 2000.0
  energy_window: 20.0
```

```bash
python discovery_runner.py --config config.yaml
```

## Examples
`python run_calc1_example.py`

## Architecture

```
CRN_interm_discovery/
├── core/          # Core abstractions (PES, data structures)
├── methods/       # Discovery methods (SSW, AFIR, MC)
├── utils/         # Utilities (I/O, relaxation, bias)
└── discovery_runner.py  # Main orchestration script
```

## Methods

### Stochastic Surface Walk (SSW)
Enhanced sampling using directional Gaussian bias potentials to escape local minima and discover new stable configurations.

### Artificial Force Induced Reaction (AFIR)  
Applies artificial forces to molecular fragments to induce chemical reactions and find new stable products.

### Monte Carlo (MC)
Statistical sampling with variants including basic MC, basin-hopping, and simulated annealing.

## Output

Results are saved as multi-frame XYZ files containing:
- Discovered molecular configurations
- Energy values and metadata
- Method and trajectory information

## Requirements

- Python ≥ 3.8
- NumPy
- ASE (Atomic Simulation Environment)
- PyYAML
- SciPy

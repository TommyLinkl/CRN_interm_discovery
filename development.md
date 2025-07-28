# Technical Development Guide

## Architecture Overview

The framework follows a modular design with clear separation of concerns:

- **Core**: Abstract interfaces and data structures
- **Methods**: Discovery algorithm implementations  
- **Utils**: Shared utilities and helper functions
- **Runners**: High-level orchestration and I/O

## Algorithm Details

### Stochastic Surface Walk (SSW)

**Core Algorithm (`methods/ssw.py`)**:

1. **Mode Generation**: Combine global thermal motion with local pairwise displacements
   ```python
   mode = global_mode + ratio_local * local_mode
   ```

2. **Dimer Rotation**: Refine displacement direction using force differences
   ```python
   dF_perp = (F2 - F1) - [(F2 - F1)·N]N
   N_new = N + γ·dF_perp  # Rotate towards instability
   ```

3. **Bias Hill Addition**: Add directional Gaussians to prevent revisiting
   ```python
   V_bias = W * exp(-proj²/(2σ²))  # proj = (R-R₀)·N
   ```

4. **Biased Relaxation**: Optimize on modified potential surface
   ```python
   V_total = V_true + V_bias
   ```

**Technical Details**:
- Uses ASE BFGS optimizer for local relaxation
- Bias hills cleared before final unbiased quench
- Metropolis acceptance with energy window filtering

### Artificial Force Induced Reaction (AFIR)

**Core Algorithm (`methods/afir.py`)**:

1. **Force Application**: Apply artificial forces between/within fragments
   ```python
   F_artificial = α * f(distance, direction)
   F_total = F_true + F_artificial
   ```

2. **Fragment Force Types**:
   - **Push**: `F = α/(r + ε) * r̂` (repulsive, 1/r decay)
   - **Pull**: `F = α * r * r̂` (attractive, linear growth)

3. **Constrained Relaxation**: Optimize under modified forces
   ```python
   # Custom ASE calculator combines true + artificial forces
   calc = AFIRCalculator(base_calc, artificial_forces)
   ```

4. **Force Removal**: Final relaxation on true potential

**Implementation Notes**:
- Artificial forces capped at `max_force` to prevent instabilities
- Default mode applies radial forces from center of mass
- Fragment specification allows targeted bond breaking/forming

### Monte Carlo Discovery

**File**: `methods/monte_carlo.py`

**Variants**:
- `MonteCarloDiscovery`: Basic MC with random atomic displacements
- `BasinHoppingMC`: Basin-hopping with larger moves + relaxation
- `SimulatedAnnealing`: Temperature-scheduled cooling

**Key Parameters**:
- `temperature`: MC temperature (1000 K)
- `step_size`: Maximum displacement per move (0.1 Å)
- `relax_frequency`: Relaxation every N steps (10)
- `adaptive_step`: Enable adaptive step sizing

## Usage Examples

### Basic Usage

```python
from core import LJPotential
from methods import StochasticSurfaceWalk
from utils import load_xyz

# Load structure
coords, atoms = load_xyz("input.xyz")

# Set up potential and method
potential = LJPotential(atoms)
ssw = StochasticSurfaceWalk(potential, atoms, temperature=2000.0)

# Run discovery
results = ssw.discover(coords, n_steps=1000)
print(f"Discovered {len(results)} configurations")
```

### Using Discovery Runner

```python
from discovery_runner import DiscoveryRunner

runner = DiscoveryRunner()

config = {
    'input_xyz': 'input.xyz',
    'methods': ['ssw', 'afir', 'mc'],
    'n_trajectories': 5,
    'n_steps_per_trajectory': 1000,
    'output_prefix': 'discovery'
}

results = runner.run_discovery(**config)
```

### Command Line Usage

```bash
python discovery_runner.py input.xyz \
    --methods ssw afir mc \
    --n-trajectories 5 \
    --n-steps 1000 \
    --temperature 1500 \
    --output-prefix my_discovery
```

## Configuration Files

The framework supports YAML and JSON configuration files:

```yaml
# config.yaml
input_xyz: "input.xyz"
methods: ["ssw", "afir", "mc"]
n_trajectories: 5
n_steps_per_trajectory: 1000
output_prefix: "discovery"
save_all_results: true
save_trajectories: true

# Method-specific parameters
ssw_params:
  ds_atom: 0.5
  W: 1.0
  NG: 8
  temperature: 2000.0
  energy_window: 20.0

afir_params:
  alpha: 0.1
  temperature: 1000.0
  max_force: 5.0

mc_params:
  temperature: 1000.0
  step_size: 0.1
  relax_frequency: 10
```

## Extension Points

### Adding New Potential Energy Surfaces

1. Inherit from `Potential` class in `core/pes.py`
2. Implement `energy()` and `forces()` methods
3. Update imports in `core/__init__.py`

```python
class MyPotential(Potential):
    def energy(self, coords: np.ndarray) -> float:
        # Your energy calculation
        return energy
    
    def forces(self, coords: np.ndarray) -> np.ndarray:
        # Your force calculation  
        return forces
```

### Adding New Discovery Methods

1. Inherit from `DiscoveryMethod` in `methods/base.py`
2. Implement `discover()` and `get_method_name()` methods
3. Update imports in `methods/__init__.py`
4. Register in `DiscoveryRunner.available_methods`

```python
class MyMethod(DiscoveryMethod):
    def get_method_name(self) -> str:
        return "MyMethod"
    
    def discover(self, initial_coords: np.ndarray, n_steps: int, **kwargs) -> ConfigurationSet:
        # Your discovery algorithm
        return results
```

## Performance Considerations

### Memory Management
- `Configuration` objects store coordinate copies
- Use `ConfigurationSet.filter_*()` methods to reduce memory usage
- Clear bias potentials when not needed (`bias.clear()`)

### Computational Efficiency
- ASE BFGS relaxation is the main bottleneck
- Adjust `fmax` and `steps` parameters for relaxation
- Use energy windows to reject high-energy configurations early
- Consider parallel trajectory execution for large runs

### Output Files
- XYZ files can become large with many configurations
- Use trajectory-specific files for better organization
- Filter results by energy before saving

## Testing and Validation

### Unit Tests
Located in `tests/` directory:
- `test_pes.py`: Potential energy surface tests
- `test_climber.py`: SSW climbing algorithm tests  
- `test_mc.py`: Monte Carlo acceptance tests
- `test_relax.py`: Local relaxation tests

### Integration Testing
Run the example script with a small test system:
```bash
python run_calc1_example.py
```

### Validation Metrics
- Energy conservation during unbiased steps
- Acceptance ratios for MC moves (target ~50%)
- Structural diversity of discovered configurations
- Comparison with known minima for test systems

## Dependencies

### Required Packages
- `numpy`: Numerical operations
- `ase`: Atomic Simulation Environment
- `pyyaml`: YAML configuration parsing
- `scipy`: Optimization algorithms (indirect via ASE)

### Installation
```bash
pip install numpy ase pyyaml scipy
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all modules are in Python path
2. **ASE Calculator Issues**: Check ASE version compatibility
3. **Memory Issues**: Reduce `n_steps` or use energy filtering
4. **Convergence Problems**: Adjust relaxation parameters
5. **File I/O Errors**: Check file permissions and paths

### Debug Mode
Set `logfile` parameter in relaxers to enable optimization logging:
```python
relaxer = LocalRelaxer(atoms, fmax=0.05, steps=500)
# Edit relaxer.py to add: dyn = BFGS(atoms, logfile='relax.log')
```

### Performance Profiling
Use Python profiler for bottleneck identification:
```bash
python -m cProfile -o profile.stats run_calc1_example.py
```

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters and returns
- Add docstrings for all public methods
- Keep line length ≤ 88 characters

### Adding Features
1. Create feature branch from main
2. Implement changes with tests
3. Update documentation
4. Submit pull request

### Reporting Issues
Include:
- Python version and OS
- ASE version  
- Input file (if possible)
- Full error traceback
- Expected vs. actual behavior

## Future Development

### Planned Features
- [ ] Parallel trajectory execution
- [ ] Advanced bias potential methods
- [ ] Integration with other ASE calculators (DFT, etc.)
- [ ] Reaction pathway analysis tools
- [ ] GUI interface for parameter tuning
- [ ] Machine learning-guided discovery

### Performance Improvements
- [ ] Cython acceleration for critical paths
- [ ] GPU acceleration for energy/force calculations
- [ ] Memory-mapped file I/O for large datasets
- [ ] Distributed computing support

### Method Extensions
- [ ] Metadynamics integration
- [ ] Replica exchange Monte Carlo
- [ ] Transition state search methods
- [ ] Free energy calculation tools
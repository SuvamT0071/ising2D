# Ising Model Simulation using Metropolis-Hastings Algorithm

## Overview
This repository contains a Python implementation of the Ising model simulation using the Metropolis-Hastings algorithm. The code allows for studying phase transitions in magnetic systems, computing specific heat, mean energy, magnetization, and magnetic susceptibility for a given lattice.

## Features
- Generate random 2D Ising spin configurations
- Compute system energy with free and periodic boundary conditions
- Simulate spin dynamics using the Metropolis-Hastings algorithm
- Compute specific heat, mean energy, magnetization, and magnetic susceptibility
- Support for periodic boundary conditions (PBC)

## Dependencies
Ensure you have the following Python libraries installed:

```bash
pip install numpy matplotlib tqdm
```

## Usage

### 1. Generate a Random Grid
```python
from isingmetro import grid_maker

grid = grid_maker(nrows=10, ncols=10)
print(grid)
```

### 2. Compute Energy
```python
from isingmetro import compute_energy

energy = compute_energy(grid)
print("Energy:", energy)
```

### 3. Run the Ising Model Simulation
```python
from isingmetro import ising_model

nsamples = 10000
temperature = 2.5
saved_energies, final_grid = ising_model(nsamples, temperature, grid)
print("Final Grid:", final_grid)
```

### 4. Compute Specific Heat
```python
from isingmetro import specific_heat

temp_range = [1.0, 2.0, 3.0, 4.0]
Cv, updated_Cv = specific_heat(grid, temp_range)
print("Specific Heat:", Cv)
```

### 5. Compute Magnetization
```python
from isingmetro import magnetize

magnetization = magnetize(grid, temp_range)
print("Magnetization:", magnetization)
```

## Periodic Boundary Conditions (PBC) Implementation
The repository also provides functions to compute system properties using periodic boundary conditions. Example:

```python
from isingmetro import compute_energy_PBC

energy_pbc = compute_energy_PBC(grid)
print("Energy with PBC:", energy_pbc)
```

## Contributing
If you'd like to contribute to improving this repository, feel free to submit a pull request.

## License
This project is licensed under the Apache license.


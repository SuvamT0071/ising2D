#load the necessary modules
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import time as t
from tqdm import tqdm

def grid_maker(nrows,ncols):
  '''
  This function generates a random classical grid of any size

  Parameters:

  - nrows: (integer) Specify the number of rows
  - ncols: (integer) Specify the number of columns

  Returns:

  The function returns your desired random grid of any size.

  '''
  if not isinstance(nrows, int) or nrows <= 0:
    raise ValueError("nrows must be a positive integer")
  if not isinstance(ncols, int) or ncols <= 0:
    raise ValueError("nrows must be a positive integer")

  grid_points = np.zeros((nrows,ncols))

  for i in range(nrows):
    for j in range(ncols):
      grid_points[i,j] = rn.choice([-1,1])

  return grid_points

def compute_energy(grid, mag = 'f'):
    """
    This function calculates the energy of a 2D square lattice with classical spins
    (Implements periodic boundary conditions).

    Parameters:
    - grid: input a 2D grid
    - mag: 'f' or 'af' to select between ferromagnetic calculation or anti-ferromagnetic calculations respectively
           (By default it is set in ferromagnetic('f') state)
    Returns:
    - Energy of the lattice
    """
    energy = 0
    nrows, ncols = grid.shape

    if mag == 'f':

        for k in range(nrows):
            for l in range(ncols):
                for dk, dl in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                    ni, nj = (k + dk) % nrows, (l + dl) % ncols #implementation of periodic boundary conditions
                    energy += -grid[k, l] * grid[ni, nj]

    elif mag == 'af':

        for k in range(nrows):
            for l in range(ncols):
                for dk, dl in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                    ni, nj = (k + dk) % nrows, (l + dl) % ncols #implementation of periodic boundary conditions
                    energy += grid[k, l] * grid[ni, nj]

    return energy / 2

def ising_model_PBC(nsamples, temperature, grid_points, mag = 'f'):
    """
    This function runs the Ising model simulation using Metropolis-Hastings.
    (Implements periodic boundary conditions)

    Parameters:
    - nsamples: Number of samples to be taken.
    - temperature: Temperature at which the system is simulated.
    - grid_points: Takes a grid of any size.
    - mag: 'f' or 'af' to select between ferromagnetic calculation or anti-ferromagnetic calculations respectively
           (By default it is set in ferromagnetic('f') state)
    Returns:
    - saved_energies: List of sampled energies by burning the first 20% of the sampled energies.
    - grid_points: The final grid after simulation.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    saved_energies = []

    nrows = grid_points.shape[0]
    ncols = grid_points.shape[1]

    for n in range(nsamples):
        i, j = np.random.randint(0, nrows), np.random.randint(0, ncols)
        temp_grid = np.copy(grid_points)
        temp_grid[i, j] = -temp_grid[i, j]

        energy = compute_energy(grid_points, mag)
        temp_energy = compute_energy(temp_grid, mag)
        energy_diff = temp_energy - energy

        p_acceptance = np.exp(-energy_diff / temperature) if temperature > 0 else 0

        if energy_diff < 0 or np.random.rand() < p_acceptance:
            grid_points[i, j] = -grid_points[i, j]
            saved_energies.append(temp_energy)
        else:
            saved_energies.append(energy)

    return saved_energies[int(nsamples/5):], grid_points

def specific_heat_PBC(grid, temp_range, nsamples=10000, mag = 'f'):
    """
    Computes specific heat over a range of temperatures.
    (Implements Periodic Boundary Conditions)

    Parameters:
    - grid: Initial 2D Ising spin configuration.
    - temp_range: List of temperatures.
    - nsamples: Number of Monte Carlo samples per temperature.
    - mag: 'f' or 'af' to select between ferromagnetic calculation or anti-ferromagnetic calculations respectively
           (By default it is set in ferromagnetic('f') state)
    Returns:
    - Cv: List of specific heat values.
    - updated_Cv: Filtered Cv list without NaN values.
    """
    energy_collections = []

    print("******************************************************")
    print("Collecting energies. Kindly wait.")
    print("****************************************************** \n")

    for t in tqdm(temp_range, desc='collecting energies', unit='iterations'):
        energy, _ = ising_model_PBC(nsamples, temperature=t, grid_points=grid, mag=mag)
        energy_collections.append(energy)

    print("******************************************************")
    print("Energy collection completed! Calculating Cv now.")
    print("****************************************************** \n")

    Cv = []
    for t, energies in zip(temp_range, energy_collections):
        var_energy = np.var(energies)
        Cv.append(var_energy / (t**2))

    print("******************************************************")
    print("Cv has been calculated. Refining it to remove NaN values now.")
    print("****************************************************** \n")

    updated_Cv = [c for c in Cv if not np.isnan(c)]

    print("******************************************************")
    print("Your results are ready!")
    print("****************************************************** \n")

    return Cv, updated_Cv

def mean_energy_PBC(grid, temp_range, nsamples=10000, mag = 'f'):
    '''
    This function calculates and gives a list of mean energy for a lattice across a given temperature range
    (Implements periodic boundary conditions)

    Parameters taken:

    - grid: Takes a 2D grid of any size.
    - temp_range: Takes a list of temperatures for which mean energy is to be calculated.
    - nsamples: How many times would you like to sample the configuration at a given temperature
                (default is 10000).
    - mag: 'f' or 'af' to select between ferromagnetic calculation or anti-ferromagnetic calculations respectively
           (By default it is set in ferromagnetic('f') state)
    Returns:

    - mean_energies: A list of calculated mean energy.
    '''
    energy_collections = []
    nrows, ncols = grid.shape
    grid = grid
    print("******************************************************")
    print("Collecting energies. Kindly wait.")
    print("****************************************************** \n")

    for t in tqdm(temp_range, desc='collecting energies', unit='temperature'):
        energy, grid = ising_model_PBC(nsamples, temperature=t, grid_points=grid, mag=mag)
        energy_collections.append(energy)

    mean_energies = []

    print("******************************************************")
    print("Energy collection completed! Calculating mean energy now.")
    print("****************************************************** \n")

    for i in tqdm(energy_collections, desc='calculating mean energy', unit='energy sample'):
        mean_energies.append(np.mean(i))

    print("******************************************************")
    print("Your results are ready!")
    print("****************************************************** \n")

    return mean_energies

def magnetize_PBC(grid, temp_range, nsweep=10000, mag='f'):
    '''
    This function calculates and gives a list of magnetization for a square lattice across a given temperature range.
    (Implements periodic boundary condition)

    Parameters:
    - grid: Takes a 2D grid of any size.
    - temp_range: List of temperatures for which magnetic susceptibility is to be calculated.
    - nsweep: Number of samples per temperature (default is 10000).
    - mag: 'f' or 'af' for ferromagnetic or anti-ferromagnetic calculations, respectively (default: 'f').

    Returns:
    - magnetization: A list of magnetization values across all temperatures.
    '''
    magnetization_values = []
    nrows, ncols = grid.shape
    N = nrows * ncols

    for T in tqdm(temp_range, desc="Processing temperatures", unit="temperature"):
        magnetizations = [] 

        for _ in range(nsweep):
            energy, grid = ising_model_PBC(nsamples=1, temperature=T, grid_points=grid, mag=mag)
            magnetizations.append(abs(np.sum(grid)) / N) 
            
        magnetization_values.append(np.mean(magnetizations))

    return magnetization_values

def mag_susceptibility_PBC(grid, temp_range, nsweep=10000, mag='f'):
    '''
    This function calculates the magnetic susceptibility for a lattice across a given temperature range.
    (Implements Periodic Boundary Conditions)
    Parameters:
    - grid: Takes a 2D grid of any size.
    - temp_range: A list of temperatures at which susceptibility is calculated.
    - nsweep: How many times would you like to sample the configuration at a given temperature
                (default is 10000).
    Returns:
    - A list of calculated magnetic susceptibilities.
    '''

    susceptibility_values = []

    print("******************************************************")
    print("Collecting magnetization data. Kindly wait.")
    print("****************************************************** \n")

    for T in tqdm(temp_range, desc="Processing temperatures", unit="temperature"):
        magnetizations = []

        for i in range(nsweep):
            energy, grid = ising_model_PBC(nsamples=1, temperature=T, grid_points=grid, mag=mag)
            magnetization = np.sum(grid)
            magnetizations.append(magnetization)

        mean_M = np.mean(magnetizations)
        mean_M2 = np.mean(np.square(magnetizations))
        chi = (mean_M2 - mean_M**2) / T
        susceptibility_values.append(chi)

    print("******************************************************")
    print("Magnetic susceptibility calculation completed!")
    print("****************************************************** \n")
    return susceptibility_values
  
def grid_maker_tri(nrows,ncols):
  '''
  This function generates a random classical triangular grid of any size

  Parameters:

  - nrows: (integer) Specify the number of rows
  - ncols: (integer) Specify the number of columns

  Returns:

  The function returns your desired random grid of any size.

  '''
  if not isinstance(nrows, int) or nrows <= 0:
    raise ValueError("nrows must be a positive integer")
  if not isinstance(ncols, int) or ncols <= 0:
    raise ValueError("nrows must be a positive integer")

  grid_points = np.zeros((nrows,ncols))

  for i in range(nrows):
    for j in range(ncols):
      grid_points[i,j] = rn.choice([-1,1])

  return grid_points

def compute_energy_triangular(grid, mag = 'f'):
    """
    This function calculates the energy of a 2D triangular lattice with classical spins
    (Implements periodic boundary conditions).

    Parameters:
    - grid: input a 2D grid
    - mag: 'f' or 'af' to select between ferromagnetic calculation or anti-ferromagnetic calculations respectively
           (By default it is set in ferromagnetic('f') state)
    Returns:
    - Energy of the lattice
    """
    energy = 0
    nrows, ncols = grid.shape

    if mag == 'f':

        for k in range(nrows):
            for l in range(ncols):
                for dk, dl in [[0, -1], [0, 1], [-1, 0], [1, 0],[1,-1],[-1,1]]:
                    ni, nj = (k + dk) % nrows, (l + dl) % ncols #implementation of periodic boundary conditions
                    energy += -grid[k, l] * grid[ni, nj]

    elif mag == 'af':

        for k in range(nrows):
            for l in range(ncols):
                for dk, dl in [[0, -1], [0, 1], [-1, 0], [1, 0],[1,-1],[-1,1]]:
                    ni, nj = (k + dk) % nrows, (l + dl) % ncols #implementation of periodic boundary conditions
                    energy += grid[k, l] * grid[ni, nj]

    return energy / 2

def ising_model_triangular(nsamples, temperature, grid_points, mag = 'f'):
    """
    This function runs the Ising model simulation using Metropolis-Hastings for a 2D triangular lattice.
    (Implements periodic boundary condition)

    Parameters:
    - nsamples: Number of samples to be taken.
    - temperature: Temperature at which the system is simulated.
    - grid_points: Takes a grid of any size.
    - mag: 'f' or 'af' to select between ferromagnetic calculation or anti-ferromagnetic calculations respectively
           (By default it is set in ferromagnetic('f') state)

    Returns:
    - saved_energies: List of sampled energies by burning the first 20% of the sampled energies.
    - grid_points: The final grid after simulation.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    saved_energies = []

    nrows, ncols = grid_points.shape

    for n in range(nsamples):
        i, j = np.random.randint(0, nrows), np.random.randint(0, ncols)
        temp_grid = np.copy(grid_points)
        temp_grid[i, j] = -temp_grid[i, j]

        energy = compute_energy_triangular(grid_points, mag)
        temp_energy = compute_energy_triangular(temp_grid, mag)
        energy_diff = temp_energy - energy

        p_acceptance = np.exp(-energy_diff / temperature) if temperature > 0 else 0

        if energy_diff < 0 or np.random.rand() < p_acceptance:
            grid_points[i, j] = -grid_points[i, j]
            saved_energies.append(temp_energy)
        else:
            saved_energies.append(energy)

    return saved_energies[int(nsamples/5):], grid_points

def specific_heat_triangular(grid, temp_range, nsamples=10000, mag = 'f'):
    """
    This function calculates and gives a list of specific heat for a lattice across a given temperature range
   (Implements periodic boundary condition)

    NOTE: Function returns 2 lists. Use 2 variables while calling the function.
    Parameters:
    - grid: Initial 2D Ising spin configuration.
    - temp_range: List of temperatures.
    - nsamples: Number of Monte Carlo samples per temperature.
    - mag: 'f' or 'af' to select between ferromagnetic calculation or anti-ferromagnetic calculations respectively
           (By default it is set in ferromagnetic('f') state)

    Returns:
    - Cv: List of specific heat values.
    - updated_Cv: Filtered Cv list without NaN values.
    """
    energy_collections = []

    print("******************************************************")
    print("Collecting energies. Kindly wait.")
    print("****************************************************** \n")

    for t in tqdm(temp_range, desc='collecting energies', unit='iterations'):
        energy, grid = ising_model_triangular(nsamples, temperature=t, grid_points=grid, mag=mag)
        energy_collections.append(energy)

    print("******************************************************")
    print("Energy collection completed! Calculating Cv now.")
    print("****************************************************** \n")

    Cv = []
    for t, energies in zip(temp_range, energy_collections):
        var_energy = np.var(energies)
        Cv.append(var_energy / (t**2))

    print("******************************************************")
    print("Cv has been calculated. Refining it to remove NaN values now.")
    print("****************************************************** \n")

    updated_Cv = [c for c in Cv if not np.isnan(c)]

    print("******************************************************")
    print("Your results are ready!")
    print("****************************************************** \n")

    return Cv, updated_Cv

def magnetize_triangular(grid, temp_range, nsweep=10000, mag='f'):
    '''
    This function calculates and gives a list of magnetization for a triangular lattice across a given temperature range.
    (Implements periodic boundary condition)

    Parameters:
    - grid: Takes a 2D grid of any size.
    - temp_range: List of temperatures for which magnetic susceptibility is to be calculated.
    - nsweep: Number of samples per temperature (default is 10000).
    - mag: 'f' or 'af' for ferromagnetic or anti-ferromagnetic calculations, respectively (default: 'f').

    Returns:
    - magnetization: A list of magnetization values across all temperatures.
    '''
    magnetization_values = []
    nrows, ncols = grid.shape
    N = nrows * ncols 

    for T in tqdm(temp_range, desc="Processing temperatures", unit="temperature"):
        magnetizations = []  

        for _ in range(nsweep):
            energy, grid = ising_model_triangular(nsamples=1, temperature=T, grid_points=grid, mag=mag)
            magnetizations.append(abs(np.sum(grid)) / N)  

        magnetization_values.append(np.mean(magnetizations))

    return magnetization_values

def mag_susceptibility_triangular(grid, temp_range, mag = 'f', nsweep=10000):
    '''
    This function calculates the magnetic susceptibility for a triangular lattice across a given temperature range.
    (Implements periodic boundary condition)
    Parameters:
    - grid: Takes a 2D grid of any size.
    - temp_range: A list of temperatures at which susceptibility is calculated.
    - mag: 'f' or 'af' to select between ferromagnetic calculation or anti-ferromagnetic calculations respectively
           (By default it is set in ferromagnetic('f') state)
    - nsweep: How many times would you like to sample the configuration at a given temperature
                (default is 10000).
    Returns:
    - A list of calculated magnetic susceptibilities.
    '''

    susceptibility_values = []

    print("******************************************************")
    print("Collecting magnetization data. Kindly wait.")
    print("****************************************************** \n")

    for T in tqdm(temp_range, desc="Processing temperatures", unit="temperature"):
        magnetizations = []

        for _ in range(nsweep):
            energy, grid = ising_model_triangular(nsamples=1, temperature=T, grid_points=grid, mag=mag)
            magnetization = np.sum(grid)
            magnetizations.append(magnetization)

        mean_M = np.mean(magnetizations)
        mean_M2 = np.mean(np.square(magnetizations))
        chi = (mean_M2 - mean_M**2) / T
        susceptibility_values.append(chi)

    print("******************************************************")
    print("Magnetic susceptibility calculation completed!")
    print("****************************************************** \n")
    return susceptibility_values

def mean_energy_triangular(grid, temp_range, nsamples=10000, mag='f'):
    '''
    This function calculates and gives a list of mean energy for a 2D triangular lattice across a given temperature range
    (Implements periodic boundary conditions)

    Parameters taken:

    - grid: Takes a 2D grid of any size.
    - temp_range: Takes a list of temperatures for which mean energy is to be calculated.
    - nsamples: How many times would you like to sample the configuration at a given temperature
                (default is 10000).
    - mag: 'f' or 'af' to select between ferromagnetic calculation or anti-ferromagnetic calculations respectively
           (By default it is set in ferromagnetic('f') state)
    Returns:

    - mean_energies: A list of calculated mean energy.
    '''
    energy_collections = []
    nrows, ncols = grid.shape
    grid = grid
    print("******************************************************")
    print("Collecting energies. Kindly wait.")
    print("****************************************************** \n")

    for t in tqdm(temp_range, desc='collecting energies', unit='temperature'):
        energy, grid = ising_model_triangular(nsamples, temperature=t, grid_points=grid, mag=mag)
        energy_collections.append(energy)

    mean_energies = []

    print("******************************************************")
    print("Energy collection completed! Calculating mean energy now.")
    print("****************************************************** \n")

    for i in tqdm(energy_collections, desc='calculating mean energy', unit='energy sample'):
        mean_energies.append(np.mean(i))

    print("******************************************************")
    print("Your results are ready!")
    print("****************************************************** \n")

    return mean_energies

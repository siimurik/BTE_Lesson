import h5py
import time as t
import numpy as np
import numba as nb
from numba import njit, prange
import matplotlib.pyplot as plt

"""
===========================================================================
 hdf2dict() function documentation
---------------------------------------------------------------------------
Essentially a modified version of the 'read_matpro()' function that 
converts all the data inside a HDF5 file from a HDF5 dict into a Python
dict.
---------------------------------------------------------------------------
Parameters:
        file_name (str): The name or path of the HDF5 file to be processed.

Returns:
    data (dict): A nested dictionary containing the datasets from the HDF5 
                file. The keys of the top-level dictionary correspond to 
                dataset names, and the values can be either nested 
                dictionaries (for struct datasets) or numpy arrays 
                (for regular datasets).

Example:
    data = hdf2dict("data.h5")
---------------------------------------------------------------------------
    Notes:
        - This function takes an HDF5 file name or path as input.
        - It reads datasets from the file and organizes them into a nested 
          dictionary structure.
        - Each dataset is represented by a key-value pair in the dictionary.
        - If a dataset is a struct (group in HDF5), it is further nested 
          within the dictionary.
        - Regular datasets are stored as numpy arrays.
===========================================================================
"""
def hdf2dict(file_name):
    # Define the dictionary to store the datasets
    data = {}
    #file_name = 'macro421_UO2_03__900K.h5'
    with h5py.File("..//02.Macro.XS.421g/" + file_name, "r") as file:
        # Iterate over the dataset names in the group
        for dataset_name in file.keys():
            # Read the dataset
            dataset = file[dataset_name]
            # Check if the dataset is a struct
            if isinstance(dataset, h5py.Group):
                # Create a dictionary to store the struct fields
                struct_data = {}
                # Iterate over the fields in the struct
                for field_name in dataset.keys():
                    # Read the field dataset
                    field_dataset = np.array(dataset[field_name])
                    # Store the field dataset in the struct dictionary
                    struct_data[field_name] = field_dataset
                # Store the struct data in the main data dictionary
                data[dataset_name] = struct_data
            else:
                # Read the dataset as a regular array
                dataset_array = np.array(dataset)
                # Store the dataset array in the dictionary
                data[dataset_name] = dataset_array
    return data

"""
========================================================
 sample_direction() function documentation
--------------------------------------------------------
Samples a random direction on the unit sphere.

Returns:
    - dirX (float): X-component of the sampled direction.
    - dirY (float): Y-component of the sampled direction.
========================================================
"""
#@nb.njit(fastmath=True)
@nb.njit
def sample_direction():
    teta = np.pi * np.random.rand()
    phi = 2.0 * np.pi * np.random.rand()
    dirX = np.sin(teta) * np.cos(phi)
    dirY = np.sin(teta) * np.sin(phi)
    return dirX, dirY

"""
===============================================================
 move_neutron() function documentation
---------------------------------------------------------------
Moves a neutron in a 2D space and ensures it stays within the 
cell boundaries.

Args:
    - x (numpy.ndarray): Array of neutron X-coordinates.
    - y (numpy.ndarray): Array of neutron Y-coordinates.
    - iNeutron (int): Index of the neutron to move.
    - pitch (float): Size of the cell in the X and Y directions.
    - freePath (float): Distance to move the neutron.
    - dirX (float): X-component of the neutron's direction.
    - dirY (float): Y-component of the neutron's direction.

Returns:
    - x (numpy.ndarray): Updated array of neutron X-coordinates.
    - y (numpy.ndarray): Updated array of neutron Y-coordinates.
===============================================================
"""
#@nb.njit(fastmath=True)
@nb.njit
def move_neutron(x, y, iNeutron, pitch, freePath, dirX, dirY):
    x[iNeutron] += freePath * dirX
    y[iNeutron] += freePath * dirY

    # If outside the cell, find the corresponding point inside the cell
    while x[iNeutron] < 0:
        x[iNeutron] += pitch
    while y[iNeutron] < 0:
        y[iNeutron] += pitch
    while x[iNeutron] > pitch:
        x[iNeutron] -= pitch
    while y[iNeutron] > pitch:
        y[iNeutron] -= pitch

    return x, y
"""
Essentialy the same as the one above, but with a different approach.
"""
#@nb.njit(fastmath=True)
@nb.njit
def move_neutronV2(x, y, iNeutron, pitch, freePath, dirX, dirY):
    x[iNeutron] += freePath * dirX
    y[iNeutron] += freePath * dirY

    # Adjust neutron positions to be within the cell boundaries
    x = np.mod(x, pitch)
    y = np.mod(y, pitch)

    return x, y

"""
==============================================================================
 calculate_cross_sections() function documentation
------------------------------------------------------------------------------
Calculates the cross sections based on the neutron's position and 
material properties.

Args:
    - fuelLeft (float): Left boundary of the fuel region.
    - fuelRight (float): Right boundary of the fuel region.
    - coolLeft (float): Left boundary of the coolant region.
    - coolRight (float): Right boundary of the coolant region.
    - x (numpy.ndarray): Array of neutron X-coordinates.
    - iNeutron (int): Index of the neutron to calculate the cross 
      sections for.
    - iGroup (numpy.ndarray): Array of neutron energy group indices.
    - fuel (dict): Dictionary containing fuel material properties.
    - cool (dict): Dictionary containing coolant material properties.
    - clad (dict): Dictionary containing cladding material properties.

Returns:
    - SigA (float): Absorption cross section.
    - SigS (numpy.ndarray): Scattering cross section as a column vector.
    - SigP (float): Probability of neutron production (fission) cross section.
==============================================================================
"""
def calculate_cross_sections(fuelLeft, fuelRight, coolLeft, coolRight, x, iNeutron, iGroup, fuel, cool, clad):
    if fuelLeft < x[iNeutron] < fuelRight:
        SigA = fuel['SigF'][iGroup[iNeutron]] + fuel['SigC'][iGroup[iNeutron]] + fuel['SigL'][iGroup[iNeutron]]
        SigS = fuel["sigS_G"]["sparse_SigS[0]"][iGroup[iNeutron], :].reshape(-1, 1)
        SigP = fuel['SigP'][0, iGroup[iNeutron]]
    elif x[iNeutron] < coolLeft or x[iNeutron] > coolRight:
        SigA = cool['SigC'][iGroup[iNeutron]] + cool['SigL'][iGroup[iNeutron]]
        SigS = cool["sigS_G"]["sparse_SigS[0]"][iGroup[iNeutron], :].reshape(-1, 1)
        SigP = 0
    else:
        SigA = clad['SigC'][iGroup[iNeutron]] + clad['SigL'][iGroup[iNeutron]]
        SigS = clad["sigS_G"]["sparse_SigS[0]"][iGroup[iNeutron], :].reshape(-1, 1)
        SigP = 0

    return SigA, SigS, SigP

@nb.njit
def calculate_cross_sections_by_region(fuelLeft, fuelRight, coolLeft, coolRight,
                                       x, iNeutron, iGroup, 
                                       fuel_SigF, fuel_SigC, fuel_SigL, fuelSigP,
                                       fuel_sparse_SigS,
                                       cool_SigC,  cool_SigL, 
                                       cool_sparse_SigS,
                                       clad_SigC, clad_SigL,
                                       clad_sparse_SigS
                                       ):
    if fuelLeft < x[iNeutron] < fuelRight:          # INPUT - Initial: 0.9 < x[iNeutron] < 2.7
        SigA = fuel_SigF[iGroup[iNeutron]] + fuel_SigC[iGroup[iNeutron]] +  fuel_SigL[iGroup[iNeutron]]
        SigS = fuel_sparse_SigS[iGroup[iNeutron], :]
        SigP = fuelSigP[0, iGroup[iNeutron]]
    elif x[iNeutron] < coolLeft or x[iNeutron] > coolRight:    # INPUT - Initial: x[iNeutron] < 0.7 or x[iNeutron] > 2.9
        SigA = cool_SigC[iGroup[iNeutron]] + cool_SigL[iGroup[iNeutron]]
        SigS = cool_sparse_SigS[iGroup[iNeutron], :]
        SigP = 0
    else:
        SigA = clad_SigC[iGroup[iNeutron]] + clad_SigL[iGroup[iNeutron]]
        SigS = clad_sparse_SigS[iGroup[iNeutron], :]
        SigP = 0

    return SigA, SigS, SigP

"""
====================================================================================
 perform_collision() function documentation
------------------------------------------------------------------------------------
Performs a collision event for a neutron based on cross section values and 
random probabilities.

Args:
    - virtualCollision (bool): Indicates whether a virtual collision occurred.
    - absorbed (bool): Indicates whether the neutron was absorbed.
    - SigS (numpy.ndarray): Array of scattering cross sections.
    - SigA (float): Absorption cross section.
    - SigP (float): Probability of neutron production (fission) cross section.
    - SigTmax (numpy.ndarray): Array of maximum total cross sections.
    - iGroup (numpy.ndarray): Array of neutron energy group indices.
    - iNeutron (int): Index of the neutron undergoing collision.
    - detectS (numpy.ndarray): Array for detecting scatterings.
    - weight (numpy.ndarray): Array of neutron weights.
    - fuel_chi (numpy.ndarray): Array of the fission energy spectrum values.

Returns:
    - absorbed (bool): Indicates whether the neutron was absorbed after the collision.
    - virtualCollision (bool): Indicates whether a virtual collision occurred.
    - iGroup (numpy.ndarray): Updated array of neutron energy group indices.
    - weight (numpy.ndarray): Updated array of neutron weights.
    - detectS (numpy.ndarray): Updated array for detecting scatterings.
====================================================================================
"""
#@nb.njit(fastmath=True)
@nb.njit
def perform_collision_v1(virtualCollision, absorbed, SigS, SigA, SigP, SigTmax,
                      iGroup, iNeutron, detectS, weight, fuel_chi):
    SigS_sum = np.sum(SigS)
    SigT = SigA + SigS_sum
    SigV = SigTmax[iGroup[iNeutron]] - SigT

    if SigV / SigTmax[iGroup[iNeutron]] >= np.random.rand():
        virtualCollision = True
    else:
        virtualCollision = False

    if SigS_sum / SigT >= np.random.rand():
        detectS[iGroup[iNeutron]] += weight[iNeutron] / SigS_sum
        iGroup[iNeutron] = np.argmax(np.cumsum(SigS) / SigS_sum >= np.random.rand())
    else:
        absorbed = True
        weight[iNeutron] *= SigP / SigA
        iGroup[iNeutron] = np.argmax(np.cumsum(fuel_chi) >= np.random.rand())

    return absorbed, virtualCollision, iGroup, weight, detectS

@nb.njit
def perform_collision(virtualCollision, absorbed, 
                      SigS, SigA, SigP, SigTmax,
                      iGroup, iNeutron, detectS, 
                      weight, fuel_chi):
    SigS_sum = np.sum(SigS)
    SigT = SigA + SigS_sum
    SigV = SigTmax[iGroup[iNeutron]] - SigT

    if SigV / SigTmax[iGroup[iNeutron]] >= np.random.rand():
        virtualCollision = True
    else:
        virtualCollision = False

        if SigS_sum / SigT >= np.random.rand():
            detectS[iGroup[iNeutron]] += weight[iNeutron] / SigS_sum
            iGroup[iNeutron] = np.argmax(np.cumsum(SigS) / SigS_sum >= np.random.rand())
        else:
            absorbed = True
            weight[iNeutron] *= SigP / SigA
            iGroup[iNeutron] = np.argmax(np.cumsum(fuel_chi) >= np.random.rand())

    return virtualCollision, absorbed, iGroup, weight, detectS

"""
=====================================================================
 russian_roulette() function documentation
--------------------------------------------------------------------
Performs the Russian roulette process to eliminate or modify neutron 
weights based on survival probabilities.

Args:
    - weight (numpy.ndarray): Array of neutron weights.
    - weight0 (numpy.ndarray): Array of initial neutron weights.

Returns:
    - weight (numpy.ndarray): Updated array of neutron weights after 
      the Russian roulette process.
=====================================================================
"""
#@nb.njit(fastmath=True)
@nb.njit
def russian_roulette(weight, weight0):
    numNeutrons = len(weight)
    for iNeutron in range(numNeutrons):
        terminateP = 1 - weight[iNeutron] / weight0[iNeutron]
        if terminateP >= np.random.rand():
            weight[iNeutron] = 0  # killed
        elif terminateP > 0:
            weight[iNeutron] = weight0[iNeutron]  # restore the weight
    return weight

"""
================================================================================
 split_neutrons() function documentation
--------------------------------------------------------------------------------
Splits neutrons into multiple new neutrons based on their weights.

Args:
    - weight (numpy.ndarray): Array of neutron weights.
    - numNeutrons (int): Number of neutrons.
    - x (numpy.ndarray): Array of neutron X-coordinates.
    - y (numpy.ndarray): Array of neutron Y-coordinates.
    - iGroup (numpy.ndarray): Array of neutron energy group indices.

Returns:
    - weight (numpy.ndarray): Updated array of neutron weights after splitting.
    - numNeutrons (int): Updated number of neutrons after splitting.
    - x (numpy.ndarray): Updated array of neutron X-coordinates after splitting.
    - y (numpy.ndarray): Updated array of neutron Y-coordinates after splitting.
    - iGroup (numpy.ndarray): Updated array of neutron energy group indices 
      after splitting.
================================================================================
"""
#@nb.njit(fastmath=True)
@nb.njit
def split_neutrons(weight, numNeutrons, x, y, iGroup):
    numNew = 0
    for iNeutron in range(numNeutrons):
        if weight[iNeutron] > 1:
            N = int(weight[iNeutron])
            if weight[iNeutron] - N > np.random.rand():
                N += 1
            weight[iNeutron] = weight[iNeutron] / N
            for iNew in range(N - 1):
                numNew += 1
                x = np.append(x, x[iNeutron])
                y = np.append(y, y[iNeutron])
                weight = np.append(weight, weight[iNeutron])
                iGroup = np.append(iGroup, iGroup[iNeutron])
    numNeutrons += numNew
    return weight, numNeutrons, x, y, iGroup

"""
===========================================================================
 update_indices() function documentation
---------------------------------------------------------------------------
Updates arrays by selecting elements based on non-zero weights.

Args:
    - x (numpy.ndarray): Array of neutron X-coordinates.
    - y (numpy.ndarray): Array of neutron Y-coordinates.
    - iGroup (numpy.ndarray): Array of neutron energy group indices.
    - weight (numpy.ndarray): Array of neutron weights.

Returns:
    -x (numpy.ndarray): Updated array of neutron X-coordinates.
    -y (numpy.ndarray): Updated array of neutron Y-coordinates.
    -iGroup (numpy.ndarray): Updated array of neutron energy group indices.
    -weight (numpy.ndarray): Updated array of neutron weights.
    -numNeutrons (int): Number of updated neutrons.
===========================================================================
"""
#@nb.njit(fastmath=True)
@nb.njit
def update_indices(x, y, iGroup, weight):
    # Get the indices of non-zero weight
    indices = np.nonzero(weight)[0]

    # Perform indexing
    x = x[indices]
    y = y[indices]
    iGroup = iGroup[indices]
    weight = weight[indices]
    
    # Update numNeutrons
    numNeutrons = weight.shape[0]
    
    return x, y, iGroup, weight, numNeutrons

"""
=======================================================================================
 calculate_keff_cycle() function documentation
---------------------------------------------------------------------------------------
Calculates the effective multiplication factor (k-eff) for a specific cycle and 
provides statistical analysis.

Args:
    - iCycle (int): Current cycle number.
    - numCycles_inactive (int): Number of inactive cycles.
    - numCycles_active (int): Number of active cycles.
    - weight (numpy.ndarray): Array of neutron weights.
    - weight0 (numpy.ndarray): Array of initial neutron weights.
    - numNeutrons (int): Number of neutrons.
    - keff_active_cycle (numpy.ndarray): Array to store k-eff values for active cycles.
    - keff_expected (numpy.ndarray): Array to store expected k-eff values.
    - sigma_keff (numpy.ndarray): Array to store standard deviations of k-eff values.

Returns:
    None (prints output messages).
=======================================================================================
"""
def calculate_keff_cycle(iCycle, numCycles_inactive, numCycles_active, weight, weight0, 
                         numNeutrons, keff_active_cycle, keff_expected, sigma_keff):
    iActive = iCycle - numCycles_inactive
    keff_cycle = np.sum(weight) / np.sum(weight0)

    if iActive <= 0:
        msg = f"Inactive cycle = {iCycle:3d}/{numCycles_inactive:3d}; k-eff cycle = {keff_cycle:.5f}; numNeutrons = {numNeutrons:3d}"
        print(msg)
    else:
        keff_active_cycle[iActive-1] = keff_cycle
        keff_expected[iActive-1] = np.mean(keff_active_cycle[:iActive])
        sigma_keff[iActive-1] = np.sqrt(np.sum((keff_active_cycle[:iActive] - keff_expected[iActive-1]) ** 2) / max(iActive - 1, 1) / iActive)

        msg = f"Active cycle = {iActive:3d}/{numCycles_active:3d}; k-eff cycle = {keff_cycle:.5f}; numNeutrons = {numNeutrons:3d}; k-eff expected = {keff_expected[iActive-1]:.5f}; sigma = {sigma_keff[iActive-1]:.5f}"
        print(msg)

"""
=============================================================================
 Documentation for the main() section of the code:
-----------------------------------------------------------------------------
 Author: Siim Erik Pugal, 2023

 The function calculates the neutron transport in a 2D (x,y) unit cell
 similar to the unit cell of the pressurized water reactor using the Monte
 Carlo method. 
-----------------------------------------------------------------------------
 This version uses the Numba boosted versions of the Cython functions written
 in 'monetpy.pyx'. Compared to MonteCarloPWR.py and mc_Cython.py, this is
 currently the faster version of the three.
-----------------------------------------------------------------------------
 Without Optimization (Pure Python):
    $ real	4m23.505s
    $ user	4m22.852s
    $ sys	0m1.205s

 After Cython Optimization
    $ real	3m28.593s
    $ user	3m28.118s
    $ sys   0m1.240s
    
 With Numba Optimization:
    $ real	1m28.181s
    $ user	1m28.348s
    $ sys	0m1.709s

=============================================================================
"""
def main():
    # Start stopwatch
    start_time = t.time()  # Placeholder for stopwatch functionality in Python

    #--------------------------------------------------------------------------
    # Number of source neutrons
    numNeutrons_born = 100      # INPUT

    # Number of inactive source cycles to skip before starting k-eff accumulation
    numCycles_inactive = 100    # INPUT

    # Number of active source cycles for k-eff accumulation
    numCycles_active = 2000     # INPUT

    # Size of the square unit cell
    pitch = 3.6  # cm           # INPUT

    # Define fuel and coolant regions
    fuelLeft, fuelRight = 0.9, 2.7  # INPUT
    coolLeft, coolRight = 0.7, 2.9  # INPUT
    #--------------------------------------------------------------------------
    # Path to macroscopic cross section data:
    # (Assuming the corresponding data files are available and accessible)
    #macro_xs_path = '..//02.Macro.XS.421g'

    # Fill the structures fuel, clad, and cool with the cross-section data
    fuel = hdf2dict('macro421_UO2_03__900K.h5')  # INPUT
    print(f"File 'macro421_UO2_03__900K.h5' has been read in.")
    clad = hdf2dict('macro421_Zry__600K.h5')     # INPUT
    print(f"File 'macro421_Zry__600K.h5' has been read in.")
    cool  = hdf2dict('macro421_H2OB__600K.h5')   # INPUT
    print(f"File 'macro421_H2OB__600K.h5' has been read in.")

    # Define the majorant: the maximum total cross-section vector
    # This part just looks at all the three different vectors and creates a new
    # vector that picks out the largest value for every index from each of the 
    # three vectros. Althought there are some rounding differences.
    SigTmax = np.max(np.vstack((fuel["SigT"], clad["SigT"], cool["SigT"])), axis=0)

    # Number of energy groups
    ng = fuel['ng']

    #--------------------------------------------------------------------------
    # Detectors
    detectS = np.zeros(ng)

    #--------------------------------------------------------------------------
    # Four main vectors describing the neutrons in a batch
    x = np.zeros(numNeutrons_born * 2)
    y = np.zeros(numNeutrons_born * 2)
    weight = np.ones(numNeutrons_born * 2)
    iGroup = np.ones(numNeutrons_born * 2, dtype=int)

    #--------------------------------------------------------------------------
    # Neutrons are assumed born randomly distributed in the cell with weight 1
    # with sampled fission energy spectrum
    numNeutrons = numNeutrons_born
    for iNeutron in range(numNeutrons):
        x[iNeutron] = np.random.rand() * pitch
        y[iNeutron] = np.random.rand() * pitch
        weight[iNeutron] = 1
        # Sample the neutron energy group
        iGroup[iNeutron] = np.argmax(np.cumsum(fuel['chi']) >= np.random.rand())
    #print(iGroup)

    #--------------------------------------------------------------------------
    # Prepare vectors for keff and standard deviation of keff
    keff_expected = np.ones(numCycles_active)
    sigma_keff = np.zeros(numCycles_active)
    keff_active_cycle = np.ones(numCycles_active)
    virtualCollision = False

    # Main (power) iteration loop
    for iCycle in range(1, numCycles_inactive + numCycles_active + 1):

        # Normalize the weights of the neutrons to make the total weight equal to
        # numNeutrons_born (equivalent to division by keff_cycle)
        #weight = (weight / np.sum(weight, axis=0, keepdims=True)) * numNeutrons_born
        weight = (weight / np.sum(weight)) * numNeutrons_born
        weight0 = weight.copy()
        #print("weight0 = ", weight0)
        #----------------------------------------------------------------------
        # Loop over neutrons
        for iNeutron in range(numNeutrons):

            absorbed = False

            #------------------------------------------------------------------
            # Neutron random walk cycle: from emission to absorption

            while not absorbed:
                #print("Neutron was absorbed!")
                # Sample free path length according to the Woodcock method
                freePath = -np.log(np.random.rand()) / SigTmax[iGroup[iNeutron]]
                #print("freePath = ", freePath)

                if not virtualCollision:
                    #print("Collision was virtual!")
                    # Sample the direction of neutron flight assuming both
                    # fission and scattering are isotropic in the lab (a strong
                    # assumption!)
                    dirX, dirY = sample_direction()
                # Fly
                x, y = move_neutron(x, y, iNeutron, pitch, freePath, dirX, dirY)

                # Find the total and scattering cross sections                    
                #SigA, SigS, SigP = calculate_cross_sections(fuelLeft, fuelRight, coolLeft, coolRight, x, iNeutron, iGroup, fuel, cool, clad)
                SigA, SigS, SigP = calculate_cross_sections_by_region(fuelLeft, fuelRight, coolLeft, coolRight,
                                                                    x, iNeutron, iGroup,
                                                                    fuel['SigF'], fuel['SigC'], fuel['SigL'], fuel['SigP'],
                                                                    fuel["sigS_G"]["sparse_SigS[0]"],
                                                                    cool['SigC'],  cool['SigL'], 
                                                                    cool["sigS_G"]["sparse_SigS[0]"],
                                                                    clad['SigC'], clad['SigL'],
                                                                    clad["sigS_G"]["sparse_SigS[0]"]
                                                                    )

                # Sample the type of the collision: virtual (do nothing) or real
                #absorbed, virtualCollision, iGroup, weight, detectS = pc_v1(absorbed, SigS, SigA, SigP, 
                #                                                                    SigTmax, iGroup, iNeutron, detectS, weight, fuel["chi"])
                virtualCollision, absorbed, iGroup, weight, detectS = perform_collision(virtualCollision, absorbed, 
                                                                                        SigS, SigA, SigP, SigTmax,
                                                                                        iGroup, iNeutron, detectS, 
                                                                                        weight, fuel['chi'])
                
            #print("virtualCollision =", virtualCollision)
            # End of neutron random walk cycle: from emission to absorption
        # End of loop over neutrons
        #-------------------------------------------------------------------------------------------
        # Russian roulette
        weight = russian_roulette(weight, weight0)

        #-------------------------------------------------------------------------------------------
        # Clean up absorbed or killed neutrons
        x, y, iGroup, weight, numNeutrons = update_indices(x, y, iGroup, weight)

        #-------------------------------------------------------------------------------------------
        # Split too "heavy" neutrons
        weight, numNeutrons, x, y, iGroup = split_neutrons(weight, numNeutrons, x, y, iGroup)

        #-------------------------------------------------------------------------------------------
        # k-eff in a cycle equals the total weight of the new generation over
        # the total weight of the old generation (the old generation weight =
        # numNeutronsBorn)
        calculate_keff_cycle(iCycle, numCycles_inactive, numCycles_active, weight, weight0, numNeutrons, keff_active_cycle, keff_expected, sigma_keff)

        # End of main (power) iteration



    # Calculate the elapsed time
    elapsed_time = t.time() - start_time

    # Create a new HDF5 file
    with h5py.File('resultsPWR.h5', 'w') as hdf:
        # Make a header for the file to be created with important parameters
        header = [
                '---------------------------------------------------------',
                'Python-based Open-source Reactor Physics Education System',
                '---------------------------------------------------------',
                '',
                'function s = resultsPWR',
                '% Results for 2D neutron transport calculation in the PWR-like unit cell using method of Monte Carlo',
                f' Number of source neutrons per k-eff cycle is {numNeutrons_born}',
                f' Number of inactive source cycles to skip before starting k-eff accumulation {numCycles_inactive}',
                f' Number of active source cycles for k-eff accumulation {numCycles_active}'
            ]

        # Write the header as attributes of the root group
        for i, line in enumerate(header):
            hdf.attrs[f'header{i}'] = line

        time = hdf.create_group("time")
        time.create_dataset("elapsedTime_(s)", data = elapsed_time)
        time.create_dataset("elapsedTime_(min)", data = elapsed_time/60)
        time.create_dataset("elapsedTime_(hrs)", data = elapsed_time/3600)

        hdf.create_dataset("keff_expected", data = keff_expected[-1])
        hdf.create_dataset("sigma", data = sigma_keff[-1])
        hdf.create_dataset("keffHistory", data = keff_expected)
        hdf.create_dataset("keffError", data = sigma_keff)

        hdf.create_dataset("eg", data = (fuel["eg"][0:ng] + fuel["eg"][1:ng+1]) / 2)

        # Calculate du and flux_du
        du = np.log(fuel["eg"][1:ng+1] / fuel["eg"][:-1])
        flux_du = detectS / du
        hdf.create_dataset("flux", data = flux_du)
        

    # Plot the k-effective
    plt.figure()
    plt.plot(keff_expected, '-r', label='k_eff')
    plt.plot(keff_expected + sigma_keff, '--b', label='k_eff +/- sigma')
    plt.plot(keff_expected - sigma_keff, '--b')
    plt.grid(True)
    plt.xlabel('Iteration number')
    plt.ylabel('k-effective')
    plt.legend()
    plt.tight_layout()
    plt.savefig('MC_01_keff.pdf')

    # Plot the spectrum
    plt.figure()
    plt.semilogx((fuel["eg"][0:ng] + fuel["eg"][1:ng+1]) / 2, flux_du)
    plt.grid(True)
    plt.xlabel('Energy, eV')
    plt.ylabel('Neutron flux per unit lethargy, a.u.')
    plt.tight_layout()
    plt.savefig('MC_02_flux_lethargy.pdf')

    # End of function

if __name__ == '__main__':
    main()
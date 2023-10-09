import re
import os
import h5py
import time
import numpy as np
import scipy as sp
from numba import njit
from pyXSteam.XSteam import XSteam

def initPWR_like():
    #global global_g
    with h5py.File("..//00.Lib/initPWR_like.h5", "w") as hdf:
        g  = hdf.create_group("g")
        th = hdf.create_group("th")
        fr = hdf.create_group("fr")

        #group.attrs["nz"] = 10
        #global_g = group

        # Input fuel rod geometry and nodalization
        g_nz = 10  # number of axial nodes
        g.create_dataset("nz", data=g_nz)

        g_fuel_rIn = 0  # inner fuel radius (m)
        g_fuel_rOut = 4.12e-3  # outer fuel radius (m)
        g_fuel_nr = 20  # number of radial nodes in fuel
        g_fuel = g.create_group("fuel")
        g_fuel.create_dataset("rIn",  data=g_fuel_rIn)
        g_fuel.create_dataset("rOut", data=g_fuel_rOut)
        g_fuel.create_dataset("nr",   data=g_fuel_nr)

        g_clad_rIn = 4.22e-3  # inner clad radius (m)
        g_clad_rOut = 4.75e-3  # outer clad radius (m)
        g_clad_nr = 5
        g_clad = g.create_group("clad")
        g_clad.create_dataset("rIn",  data=g_clad_rIn)
        g_clad.create_dataset("rOut", data=g_clad_rOut)
        g_clad.create_dataset("nr",   data=g_clad_nr)

        g_cool_pitch = 13.3e-3  # square unit cell pitch (m)
        g_cool_rOut = np.sqrt(g_cool_pitch**2 / np.pi)  # equivalent radius of the unit cell (m)
        g_cool = g.create_group("cool")
        g_cool.create_dataset("pitch", data=g_cool_pitch)
        g_cool.create_dataset("rOut",  data=g_cool_rOut)

        g_dz0 = 0.3 * np.ones(g_nz)  # height of the node (m)
        g_dzGasPlenum = 0.2  # height of the fuel rod gas plenum assuming it is empty (m)
        g.create_dataset("dz0", data = g_dz0)
        g.create_dataset("dzGasPlenum", data = g_dzGasPlenum)

        # Input average power rating in fuel
        th_qLHGR0 = np.array([  [0, 10, 1e20],  # time (s)
                                [200e2, 200e2, 200e2]   ])  # linear heat generation rate (W/m)
        th.create_dataset("qLHGR0", data = th_qLHGR0)

        # Input fuel rod parameters
        fr_clad_fastFlux = np.array([   [0, 10, 1e20],  # time (s)
                                        [1e13, 1e13, 1e13]  ])  # fast flux in cladding (1/cm2-s)
        fr_clad = fr.create_group("clad")
        fr_clad.create_dataset("fastFlux", data = fr_clad_fastFlux)

        fr_fuel_FGR = np.array([[0, 10, 1e20],  # time (s)
                                [0.03, 0.03, 0.03]])  # fission gas release (-)
        fr_fuel = fr.create_group("fuel")
        fr_fuel.create_dataset("FGR", data = fr_fuel_FGR)

        fr_ingas_Tplenum = 533  # fuel rod gas plenum temperature (K)
        fr_ingas_p0 = 1  # as-fabricated helium pressure inside fuel rod (MPa)
        fr_fuel_por = 0.05 * np.ones((g_nz, g_fuel_nr))  # initial fuel porosity (-)
        fr_ingas = fr.create_group("ingas")
        fr_ingas.create_dataset("Tplenum", data = fr_ingas_Tplenum)
        fr_ingas.create_dataset("p0",      data = fr_ingas_p0)
        fr_fuel.create_dataset("por",      data = fr_fuel_por)

        # Input channel geometry
        g_aFlow = 8.914e-5 * np.ones(g_nz)  # flow area (m2)
        g.create_dataset("aFlow", data = g_aFlow)

        # Input channel parameters
        th_mdot0_ = np.array([  [0, 10, 1000],  # time (s)
                                [0.3, 0.3, 0.3]])  # flowrate (kg/s) 0.3
        th_p0 = 16  # coolant pressure (MPa)
        th_T0 = 533.0  # inlet temperature (K)
        th.create_dataset("mdot0", data = th_mdot0_)
        th.create_dataset("p0",    data = th_p0)
        th.create_dataset("T0",    data = th_T0)

        # Initialize fuel geometry
        g_fuel_dr0 = (g_fuel_rOut - g_fuel_rIn) / (g_fuel_nr - 1)  # fuel node radial thickness (m)
        g_fuel_r0 = np.arange(g_fuel_rIn, g_fuel_rOut + g_fuel_dr0, g_fuel_dr0)  # fuel node radius (m)
        g_fuel_r0_ = np.concatenate(([g_fuel_rIn], np.interp(np.arange(1.5, g_fuel_nr + 0.5), np.arange(1, g_fuel_nr + 1),
                                                                    g_fuel_r0), [g_fuel_rOut]))  # fuel node boundary (m)
        g_fuel_a0_ = np.transpose(np.tile(2*np.pi*g_fuel_r0_[:, None], g_nz)) * np.tile(g_dz0[:, None], (1, g_fuel_nr + 1))  # XS area of fuel node boundary (m2)
        g_fuel_v0 = np.transpose(np.tile(np.pi*np.diff(g_fuel_r0_**2)[:, None], g_nz)) * np.tile(g_dz0[:, None], (1, g_fuel_nr))  # fuel node volume (m3)
        g_fuel_vFrac = (g_fuel_rOut**2 - g_fuel_rIn**2) / g_cool_rOut**2
        g_fuel.create_dataset("dr0",   data = g_fuel_dr0)
        g_fuel.create_dataset("r0",    data = g_fuel_r0)
        g_fuel.create_dataset("r0_",   data = g_fuel_r0_)
        g_fuel.create_dataset("a0_",   data = g_fuel_a0_)
        g_fuel.create_dataset("v0",    data = g_fuel_v0)
        g_fuel.create_dataset("vFrac", data = g_fuel_vFrac)

        # Initialize clad geometry
        g_clad_dr0 = (g_clad_rOut - g_clad_rIn) / (g_clad_nr - 1)  # clad node radial thickness (m)
        g_clad_r0 = np.arange(g_clad_rIn, g_clad_rOut + g_clad_dr0, g_clad_dr0)  # clad node radius (m)
        g_clad_r0_ = np.concatenate(([g_clad_rIn], np.interp(np.arange(1.5, g_clad_nr + 0.5), np.arange(1, g_clad_nr + 1), 
                                                                    g_clad_r0), [g_clad_rOut]))  # clad node boundary (m)
        g_clad_a0_ = np.transpose(np.tile(2 * np.pi * g_clad_r0_[:, None], g_nz)) * np.tile(g_dz0[:, None], (1, g_clad_nr + 1))  # XS area of clad node boundary (m2)
        g_clad_v0 = np.transpose(np.tile(np.pi*np.diff(g_clad_r0_**2)[:, None], g_nz)) * np.tile(g_dz0[:, None], (1, g_clad_nr))  # clad node volume (m3)
        g_clad_vFrac = (g_clad_rOut**2 - g_clad_rIn**2) / g_cool_rOut**2
        g_clad.create_dataset("dr0",   data = g_clad_dr0)
        g_clad.create_dataset("r0",    data = g_clad_r0)
        g_clad.create_dataset("r0_",   data = g_clad_r0_)
        g_clad.create_dataset("a0_",   data = g_clad_a0_)
        g_clad.create_dataset("v0",    data = g_clad_v0)
        g_clad.create_dataset("vFrac", data = g_clad_vFrac)

        # Initialize gap geometry
        dimensions = tuple(range(1, g_nz+1))
        g_gap_dr0 = (g_clad_rIn - g_fuel_rOut) * np.ones(dimensions)   # initial cold gap (m)
        g_gap_r0_ = (g_clad_rIn + g_fuel_rOut) / 2  # average gap radius (m)
        g_gap_a0_ = (2 * np.pi * g_gap_r0_ * np.ones((g_nz, 1))) * g_dz0  # XS area of the mid-gap (m2)
        g_gap_vFrac = (g_clad_rIn**2 - g_fuel_rOut**2) / g_cool_rOut**2
        g_gap = g.create_group("gap")
        g_gap.create_dataset("dr0",   data = g_gap_dr0.flatten())
        g_gap.create_dataset("r0_",   data = g_gap_r0_)
        g_gap.create_dataset("a0_",   data = g_gap_a0_)
        g_gap.create_dataset("vFrac", data = g_gap_vFrac)

        # Initialize as-fabricated inner volumes and gas amount
        g_vGasPlenum = g_dzGasPlenum * np.pi * g_clad_rIn**2  # gas plenum volume (m3)
        g_vGasGap = g_dz0 * np.pi * (g_clad_rIn**2 - g_fuel_rOut**2)  # gas gap volume (m3)
        g_vGasCentralVoid = g_dz0 * np.pi * g_fuel_rIn**2  # gas central void volume (m3)
        fr_ingas_muHe0 = fr_ingas_p0 * (g_vGasPlenum + np.sum(g_vGasGap + g_vGasCentralVoid)) / (8.31e-6 * 293)  # as-fabricated gas amount inside fuel rod (mole)
        g.create_dataset("vGasPlenum",      data = g_vGasPlenum)
        g.create_dataset("vGasGap",         data = g_vGasGap)
        g.create_dataset("vGasCentralVoid", data = g_vGasCentralVoid)
        fr_ingas.create_dataset("muHe0",    data = fr_ingas_muHe0)

        # Initialize gas gap status
        g_gap_open = np.ones(g_nz)
        g_gap_clsd = np.zeros(g_nz)
        g_gap.create_dataset("open", data = g_gap_open)
        g_gap.create_dataset("clsd", data = g_gap_clsd)

        # Initialize fuel and clad total deformation components
        fr_fuel_eps0 = np.zeros((3, g_nz, g_fuel_nr))
        fr_clad_eps0 = np.zeros((3, g_nz, g_clad_nr))
        fuel_eps0 = fr_fuel.create_group("eps0")
        clad_eps0 = fr_clad.create_group("eps0")
        for i in range(3):
            fr_fuel_eps0[i] = np.zeros((g_nz, g_fuel_nr))
            fr_clad_eps0[i] = np.zeros((g_nz, g_clad_nr))
            fuel_eps0.create_dataset(f"eps0(0,{i})", data = fr_fuel_eps0[i])
            clad_eps0.create_dataset(f"eps0(0,{i})", data = fr_clad_eps0[i])

        # Initialize flow channel geometry
        g_volFlow = g_aFlow * g_dz0  # volume of node (m3)
        g_areaHX = 2 * np.pi * g_clad_rOut * g_dz0  # heat exchange area(m2)(m2)
        g_dHyd = 4 * g_volFlow / g_areaHX  # hydraulic diameter (m)
        g_cool_vFrac = (g_cool_rOut**2 - g_clad_rOut**2) / g_cool_rOut**2
        g.create_dataset("volFlow",    data = g_volFlow)
        g.create_dataset("areaHX",     data = g_areaHX)
        g.create_dataset("dHyd",       data = g_dHyd)
        g_cool.create_dataset("vFrac", data = g_cool_vFrac)

        # Initialize thermal hydraulic parameters
        # Path to steam-water properties
        steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
        th_h0 = steamTable.h_pt(th_p0 / 10, th_T0 - 273) * 1e3  # water enthalpy at core inlet (J/kg)
        th_h = np.ones(g_nz) * th_h0  # initial enthalpy in nodes (kJ/kg)
        th_p = np.ones(g_nz) * th_p0  # initial pressure in nodes (MPa)
        th.create_dataset("h0", data = th_h0)
        th.create_dataset("h", data = th_h)
        th.create_dataset("p", data = th_p)

def readPWR_like(input_keyword):
    # Define the mapping of input keyword to group name
    group_mapping = {
        "fr": "fr",
        "g": "g",
        "th": "th"
    }
    # Define the dictionary to store the datasets
    data = {}
    # Open the HDF5 file
    with h5py.File("..//00.Lib/initPWR_like.h5", "r") as file:
        # Get the group name based on the input keyword
        group_name = group_mapping.get(input_keyword)
        if group_name is not None:
            # Get the group
            group = file[group_name]
            # Iterate over the dataset names in the group
            for dataset_name in group.keys():
                # Read the dataset
                dataset = group[dataset_name]
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

    # Access the datasets by their names
    #print(data.keys())
    return data

def prep2D(group):
    subgroups_data = []

    def get_data(name, obj):
        if isinstance(obj, h5py.Dataset):
            subgroup_data = np.array(obj)
            subgroups_data.append(subgroup_data)

    group.visititems(get_data)
    return np.array(subgroups_data) 

#@njit
def prepareIntoND(*matrices):
    num_cells = len(matrices)
    num_rows_per_cell = [matrix.shape[0] for matrix in matrices]
    num_cols = matrices[0].shape[1]

    # Create an empty 3D array with the desired shape
    result3D = np.zeros((num_cells, max(num_rows_per_cell), num_cols))

    # Fill the 3D array with data from the 2D matrices
    for cell in range(num_cells):
        num_rows = num_rows_per_cell[cell]
        result3D[cell, :num_rows, :] = matrices[cell]

    return result3D

def sigmaZeros(sigTtab, sig0tab, aDen, SigEscape):
    # Number of energy groups
    ng = 421

    # Define number of isotopes in the mixture
    nIso = len(aDen)

    # Define the size of sigT and a temporary value 
    # named sigT_tmp for interpolation values
    sigT = np.zeros((nIso, ng))
    sigT_tmp = 0

    # first guess for sigma-zeros is 1e10 (infinite dilution)
    sig0 = np.ones((nIso, ng)) * 1e10

    # Loop over energy group
    for ig in range(ng):
        # Error to control sigma-zero iterations
        err = 1e10
        nIter = 0

        # sigma-sero iterations until the error is below selected tolerance (1e-6)
        while err > 1e-6:
            # Loop over isotopes
            for iIso in range(nIso):
                # Find cross section for the current sigma-zero by interpolating
                # in the table
                if np.count_nonzero(sig0tab[iIso]) == 1:
                    sigT[iIso, ig] = sigTtab[iIso][0, ig]
                else:
                    log10sig0 = np.minimum(10, np.maximum(0, np.log10(sig0[iIso, ig])))
                    sigT_tmp = sp.interpolate.interp1d( np.log10(sig0tab[iIso][np.nonzero(sig0tab[iIso])]), 
                                                    sigTtab[iIso][:, ig][np.nonzero(sigTtab[iIso][:, ig])], 
                                                    kind='linear')(log10sig0)
                    sigT[iIso, ig] = sigT_tmp
                #sigT = sigT.item() 
                #sigTtab[iIso][np.isnan(sigTtab[iIso])] = 0  # not sure if 100% necessary, but a good mental check

            err = 0
            # Loop over isotopes
            for iIso in range(nIso):
                # Find the total macroscopic cross section for the mixture of
                # the background isotopes
                summation = 0
                # Loop over background isotopes
                for jIso in range(nIso):
                    if jIso != iIso:
                        summation += sigT[jIso, ig] * aDen[jIso]

                tmp = (SigEscape + summation) / aDen[iIso]
                err += (1 - tmp / sig0[iIso, ig])**2
                sig0[iIso, ig] = tmp

            err = np.sqrt(err)
            nIter += 1
            if nIter > 100:
                print('Error: too many sigma-zero iterations.')
                return

    return sig0

@njit
def interp1d_numba(x, y, x_new):
    """
    Linear interpolation function using NumPy and Numba.
    
    Parameters:
        x (array-like): 1-D array of x-coordinates of data points.
        y (array-like): 1-D array of y-coordinates of data points.
        x_new (array-like): 1-D array of x-coordinates for which to interpolate.
        
    Returns:
        array-like: 1-D array of interpolated values corresponding to x_new.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_new = np.asarray(x_new)
    
    # Sorting based on x
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Handle boundary cases
    if x_new < x_sorted[0]:
        return y_sorted[0]
    elif x_new > x_sorted[-1]:
        return y_sorted[-1]
    
    # Find indices of the nearest points for interpolation
    idx = np.searchsorted(x_sorted, x_new, side='right') - 1
    idx = np.maximum(0, np.minimum(len(x_sorted) - 2, idx))
    
    # Compute weights for interpolation
    x0 = x_sorted[idx]
    x1 = x_sorted[idx + 1]
    y0 = y_sorted[idx]
    y1 = y_sorted[idx + 1]
    weights = (x_new - x0) / (x1 - x0)
    
    # Perform linear interpolation
    interpolated_values = y0 + (y1 - y0) * weights
    return interpolated_values

def interpSigS(jLgn, element, temp, Sig0):
    # Number of energy groups
    ng = 421
    elementDict = {
    'H01':  'H_001',
    'O16':  'O_016',
    'U235': 'U_235',
    'U238': 'U_238',
    'O16':  'O_016',
    'ZR90': 'ZR090',
    'ZR91': 'ZR091',
    'ZR92': 'ZR092',
    'ZR94': 'ZR094',
    'ZR96': 'ZR096',
    'B10':  'B_010',
    'B11':  'B_011'
    }
    # Path to microscopic cross section data:
    micro_XS_path = '../01.Micro.XS.421g'
    # Open the HDF5 file based on the element
    filename = f"micro_{elementDict[element]}__{temp}K.h5"
    with h5py.File(micro_XS_path + '/' + filename, 'r') as f:
        s_sig0 = np.array(f.get('sig0_G').get('sig0'))
        findSigS = list(f.get('sigS_G').items())
        string = findSigS[-1][0]  # 'sigS(2,5)'

        # Extract numbers using regular expression pattern
        pattern = r"sigS\((\d+),(\d+)\)"
        match = re.search(pattern, string)

        if match:
            x_4D = int(match.group(1)) + 1
            y_4D = int(match.group(2)) + 1
        else:
            print("No match found.")

        # Create the empty 3D numpy array
        s_sigS = np.zeros((x_4D, y_4D, ng, ng))

        # Access the data from the subgroups and store it in the 3D array
        for i in range(x_4D):
            for j in range(y_4D):
                dataset_name = f'sigS({i},{j})'
                s_sigS[i, j] = np.array(f.get('sigS_G').get(dataset_name))
                
        # Number of sigma-zeros
        nSig0 = len(s_sig0)

        if nSig0 == 1:
            sigS = s_sigS[jLgn][0]
        else:
            tmp1 = np.zeros((nSig0, sp.sparse.find(s_sigS[jLgn][0])[2].shape[0]))
            for iSig0 in range(nSig0):
                ifrom, ito, tmp1[iSig0, :] = sp.sparse.find(s_sigS[jLgn][iSig0])

            # Number of non-zeros in a scattering matrix
            nNonZeros = tmp1.shape[1]
            tmp2 = np.zeros(nNonZeros)
            for i in range(nNonZeros):
                # Numpy method
                #log10sig0 = min(10, max(0, np.log10(Sig0[ifrom[i]])))
                #tmp2[i] = np.interp(np.log10(log10sig0), np.log10(s_sig0), tmp1[:, i])
                
                # SciPy method
                #log10sig0 = np.log10(Sig0[ifrom[i]])
                #log10sig0 = min(1, max(0, log10sig0))
                #interp_func = sp.interpolate.interp1d(np.log10(s_sig0), tmp1[:, i])
                #tmp2[i] = interp_func(log10sig0)

                # Numba method
                log10sig0 = np.log10(Sig0[ifrom[i]])
                log10sig0 = min(1, max(0, log10sig0))
                tmp2[i] = interp1d_numba(np.log10(s_sig0), tmp1[:, i], log10sig0)

            sigS = sp.sparse.coo_matrix((tmp2, (ifrom, ito)), shape=(ng, ng)).toarray()

    return sigS

@njit
def numba_sparse_find(matrix):
    rows, cols = [], []
    values = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                rows.append(i)
                cols.append(j)
                values.append(matrix[i, j])
    return np.array(rows), np.array(cols), np.array(values)

@njit
def numba_coo_matrix(tmp2, ifrom, ito, shape):
    coo_matrix = np.zeros(shape)
    for i in range(len(ifrom)):
        coo_matrix[ifrom[i], ito[i]] = tmp2[i]
    return coo_matrix

@njit
def numba_prep_interpSigS(jLgn, s_sig0, s_sigS, Sig0):
    # Number of sigma-zeros
    nSig0 = len(s_sig0)
    if nSig0 == 1:
        sigS = s_sigS[jLgn][0]
    else:
        tmp1 = np.zeros((nSig0, numba_sparse_find(s_sigS[jLgn][0])[2].shape[0]))
        for iSig0 in range(nSig0):
            ifrom, ito, tmp1[iSig0, :] = numba_sparse_find(s_sigS[jLgn][iSig0])

        # Number of non-zeros in a scattering matrix
        nNonZeros = tmp1.shape[1]
        tmp2 = np.zeros(nNonZeros)
        for i in range(nNonZeros):
            # Numpy method
            #log10sig0 = min(10, max(0, np.log10(Sig0[ifrom[i]])))
            #tmp2[i] = np.interp(np.log10(log10sig0), np.log10(s_sig0), tmp1[:, i])
            # Numba method
            log10sig0 = np.log10(Sig0[ifrom[i]])
            log10sig0 = min(1, max(0, log10sig0))
            tmp2[i] = interp1d_numba(np.log10(s_sig0), tmp1[:, i], log10sig0)
        shape = (421,421)
        sigS = numba_coo_matrix(tmp2, ifrom, ito, shape)

    return sigS

def boosted_interpSigS(jLgn, element, temp, Sig0):
    # Number of energy groups
    ng = 421
    elementDict = {
    'H01':  'H_001',
    'O16':  'O_016',
    'U235': 'U_235',
    'U238': 'U_238',
    'O16':  'O_016',
    'ZR90': 'ZR090',
    'ZR91': 'ZR091',
    'ZR92': 'ZR092',
    'ZR94': 'ZR094',
    'ZR96': 'ZR096',
    'B10':  'B_010',
    'B11':  'B_011'
    }
    # Path to microscopic cross section data:
    micro_XS_path = '../01.Micro.XS.421g' 
    # Open the HDF5 file based on the element
    filename = f"micro_{elementDict[element]}__{temp}K.h5"
    with h5py.File(micro_XS_path + '/' + filename, 'r') as f:
        s_sig0 = np.array(f.get('sig0_G').get('sig0'))
        findSigS = list(f.get('sigS_G').items())
        string = findSigS[-1][0]  # 'sigS(2,5)'

        # Extract numbers using regular expression pattern
        pattern = r"sigS\((\d+),(\d+)\)"
        match = re.search(pattern, string)

        if match:
            x_4D = int(match.group(1)) + 1
            y_4D = int(match.group(2)) + 1
        else:
            print("No match found.")

        # Create the empty 3D numpy array
        s_sigS = np.zeros((x_4D, y_4D, ng, ng))

        # Access the data from the subgroups and store it in the 3D array
        for i in range(x_4D):
            for j in range(y_4D):
                dataset_name = f'sigS({i},{j})'
                s_sigS[i, j] = np.array(f.get('sigS_G').get(dataset_name))
                
        sigS = numba_prep_interpSigS(jLgn, s_sig0, s_sigS, Sig0)

    return sigS

def writeMacroXS(s_struct, matName):
    print(f'Write macroscopic cross sections to the file: {matName}.h5')
    
    # Convert int and float to np.ndarray
    for key in s_struct.keys():
        data = s_struct[key]
        if isinstance(data, (int, float)):
            s_struct[key] = np.array(data)

    # Create the output HDF5 file
    with h5py.File(matName + '.h5', 'w') as f:
        # Make a header for the file to be created with important parameters
        header = [
            '---------------------------------------------------------',
            'Python-based Open-source Reactor Physics Education System',
            '---------------------------------------------------------',
            'Author: Siim Erik Pugal',
            '',
            'Macroscopic cross sections for water solution of boric acid',
            f'Water temperature:    {s_struct["temp"]:.1f} K',
            f'Water pressure:       {s_struct["p"]:.1f} MPa',
            f'Water density:        {s_struct["den"]:.5f} g/cm3',
            f'Boron concentration:  {s_struct["bConc"]*1e6:.1f} ppm'
        ]

        # Write the header as attributes of the root group
        for i, line in enumerate(header):
            f.attrs[f'header{i}'] = line
            
        # Convert non-array values to np.ndarray and write as datasets
        for key in s_struct.keys():
            data = s_struct[key]
            if isinstance(data, np.ndarray):
                f.create_dataset(key, data=data)
            elif isinstance(data, list):
                f.create_dataset(key, data=data)
            elif isinstance(data, dict):
                group = f.create_group(key)
                for subkey, subdata in data.items():
                    group.create_dataset(subkey, data=subdata)
    
        # Rest of the code remains unchanged
        s_SigS = np.zeros((3, 421, 421))
        for i in range(3):
            s_SigS[i] = s_struct['SigS'][f'SigS[{i}]']

        SigS_G = f.create_group("sigS_G")
        ng = s_struct['ng']
        for j in range(s_SigS.shape[0]):
            Sig = np.zeros(sp.sparse.find(s_SigS[j])[2].shape[0])
            ito, ifrom, Sig = sp.sparse.find(s_SigS[j])
            sigS_sparse = sp.sparse.coo_matrix((Sig, (ifrom, ito)), shape=(ng, ng))
            sigS_new = sigS_sparse.toarray()
            SigS_G.create_dataset(f"Sig[{j}]", data=Sig)
            SigS_G.create_dataset(f"sparse_SigS[{j}]", data=sigS_new)
        SigS_G.attrs['description'] = f'Scattering matrix for {s_SigS.shape[0]} Legendre components'
        SigS_G.create_dataset("ifrom", data=ifrom)
        SigS_G.create_dataset("ito", data=ito)

        # Delete a group
        if 'SigS' in f:
            del f['SigS']

        s_Sig2 = s_struct['Sig2']
        Sig2_G = f.create_group("sig2_G")
        Sig = np.zeros(sp.sparse.find(s_Sig2)[2].shape[0])
        ito, ifrom, Sig = sp.sparse.find(s_Sig2)
        sigS_sparse = sp.sparse.coo_matrix((Sig, (ifrom, ito)), shape=(ng, ng))
        sigS_new = sigS_sparse.toarray()
        Sig2_G.attrs['description'] = 'Python-based Neutron Transport Simulation'
        Sig2_G.create_dataset("Sig", data=Sig)
        Sig2_G.create_dataset("sparse_Sig2", data=sigS_new)
        Sig2_G.create_dataset("ifrom", data=ifrom)
        Sig2_G.create_dataset("ito", data=ito)

        # Delete a dataset
        if 'Sig2' in f:
            del f['Sig2']
        if "p" in f:    
            del f["p"]
        if "Uconc" in f:
            del f["Uconc"]

        f.create_dataset('fissile', data=1)
        if np.all(s_struct['SigP'][0] == 0):
            if 'fissile' in f:
                del f['fissile']
            f.create_dataset('fissile', data=0)
            if 'SigF' in f:
                del f['SigF']
            f.create_dataset('SigF', data=np.zeros((1, ng)))
            if 'SigP' in f:
                del f['SigP']
            f.create_dataset('SigP', data=np.zeros((1, ng)))
            if 'chi' in f:
                del f['chi']
            f.create_dataset('chi', data=np.zeros((1, ng)))

    print('Done.')


def main():
    """
    =========================================================================
    Documentation for the main() section of the code:
    -------------------------------------------------------------------------
    Author: Siim Erik Pugal, 2023

    The function reads the MICROscopic group cross sections in the HDF5
    format and calculates from them the MACROscopic cross sections for water
    solution of boric acid which is similar to the coolant of the pressurized
    water reactor.
    =========================================================================
    """
    # number of energy groups
    ng = 421

    # Boron is composed of two stable isotopes: B10 and B11 with the following
    # molar fractions:
    molFrB = np.array([0.199, 0.801])

    # Input and initialize the geometry of the PWR-like unit cell (the function
    # is in '../00.Lib')
    lib_path = os.path.join('..', '00.Lib')
    file_path = os.path.join(lib_path, 'initPWR_like.h5')

    if not os.path.exists(file_path):
        # File doesn't exist, call initPWR_like() function
        initPWR_like()

    # Path to microscopic cross section data:
    micro_XS_path = '../01.Micro.XS.421g'                                # INPUT

    # # Load HDF5 files for H2O and B isotopes
    hdf5_H01 = h5py.File(micro_XS_path + '/micro_H_001__600K.h5', 'r')  # INPUT
    print(f"File 'micro_H_001__600K.h5' has been read in.")
    hdf5_O16 = h5py.File(micro_XS_path + '/micro_O_016__600K.h5', 'r')  # INPUT
    print(f"File 'micro_O_016__600K.h5' has been read in.")
    hdf5_B10 = h5py.File(micro_XS_path + '/micro_B_010__600K.h5', 'r')  # INPUT
    print(f"File 'micro_B_010__600K.h5' has been read in.")
    hdf5_B11 = h5py.File(micro_XS_path + '/micro_B_011__600K.h5', 'r')  # INPUT
    print(f"File 'micro_B_011__600K.h5' has been read in.")

    H2OB = {}

    # Set input parameters
    H2OB['temp'] = 600  # K
    H2OB['p'] = 16  # MPa
    H2OB['bConc'] = 4000e-6  # 1e-6 = 1 ppm
    H2OB['eg'] = np.array(hdf5_H01.get('en_G').get('eg'))  # Assuming 'eg' is the dataset name
    H2OB['ng'] = 421

    # Mass of one "average" H2OB molecule in atomic unit mass [a.u.m.]:
    H2OB['aw'] = 2 * hdf5_H01.attrs.get('aw') + hdf5_O16.attrs.get('aw') + \
                 H2OB['bConc'] * (molFrB[0] * hdf5_B10.attrs.get('aw') + 
                                  molFrB[1] * hdf5_B11.attrs.get('aw'))

    # The function returns water density at specified pressure (MPa) and temperature (C):
    steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
    density = steamTable.rho_pt(H2OB['p'] * 10, H2OB['temp'] - 273)

    # The water density:
    H2OB['den'] = density * 1e-3  # [g/cm3]
    rho = H2OB['den'] * 1.0e-24  # [g/(barn*cm)]
    rho = rho / 1.660538e-24  # [(a.u.m.)/(barn*cm)]
    rho = rho / H2OB['aw']  # [number of H2O molecules/(barn*cm)]

    # The names of isotopes
    H2OB['isoName'] = ['H01', 'O16', 'B10', 'B11']

    # The number densities of isotopes:
    H2OB['numDen'] = np.array([2 * rho, rho, rho * H2OB['bConc'] * molFrB[0], 
                            rho * H2OB['bConc'] * molFrB[1]])

    # Prepare for sigma-zero iterations:
    sigTtab = prepareIntoND(
                prep2D(hdf5_H01.get('sigT_G')), 
                prep2D(hdf5_O16.get('sigT_G')), 
                prep2D(hdf5_B10.get('sigT_G')), 
                prep2D(hdf5_B11.get('sigT_G'))
            )

    sig0_sizes = []
    for i in range(len(H2OB["isoName"])):
        sig0_sizes.append(len(np.array(eval(f'hdf5_{H2OB["isoName"][i]}').get('sig0_G').get('sig0'))))

    sig0tab = np.zeros((len(H2OB["isoName"]), max(sig0_sizes)))
    for i, size in enumerate(sig0_sizes):
        #print("i =", i, "size =", size)
        sig0tab[i, :size] = np.array(eval(f'hdf5_{H2OB["isoName"][i]}').get('sig0_G').get('sig0'))

    aDen = H2OB['numDen']

    # SigEscape -- escape cross section, for simple convex objects (such as
    # plates, spheres, or cylinders) is given by S/(4V), where V and S are the
    # volume and surface area of the object, respectively
    SigEscape = 0

    print('Sigma-zero iterations.')
    H2OB['sig0'] = sigmaZeros(sigTtab, sig0tab, aDen, SigEscape)
    print('Done.')

    print('Interpolation of microscopic cross sections for the found sigma-zeros. ')
    sigCtab = prepareIntoND(
                prep2D(hdf5_H01.get('sigC_G')), 
                prep2D(hdf5_O16.get('sigC_G')), 
                prep2D(hdf5_B10.get('sigC_G')), 
                prep2D(hdf5_B11.get('sigC_G'))
            )

    sigL_sizes = []
    for i in range(len(H2OB["isoName"])):
        sigL_data = np.array(eval(f'hdf5_{H2OB["isoName"][i]}').get('sigL_G').get('sigL'))
        sigL_sizes.append(len(sigL_data))

    maxL_size = max(sigL_sizes)
    sigLtab = np.zeros((len(H2OB["isoName"]), maxL_size, H2OB['ng']))
    col_start = 0
    for i, size in enumerate(sigL_sizes):
        sigL_data = np.array(eval(f'hdf5_{H2OB["isoName"][i]}').get('sigL_G').get('sigL'))
        sigLtab[i, :size, col_start:col_start+421] = sigL_data.reshape(size, 421)

    sigC, sigL = np.zeros((len(aDen), H2OB['ng'])), np.zeros((len(aDen), H2OB['ng']))

    for ig in range(H2OB['ng']):
        # Number of isotopes in the mixture
        nIso = len(aDen)
        # Loop over isotopes
        for iIso in range(nIso):
            # Find cross sections for the found sigma-zeros
            #if len(sig0tab[iIso][np.nonzero(sig0tab[iIso])]) == 1:
            if np.count_nonzero(sig0tab[iIso]) == 1:
                sigC[iIso, ig] = sigCtab[iIso][0, ig]
                sigL[iIso, ig] = sigLtab[iIso][0, ig]
            else:
                log10sig0 = min(10, max(0, np.log10(H2OB['sig0'][iIso, ig])))
                arrayLength = len(sig0tab[iIso][np.nonzero(sig0tab[iIso])])
                x = np.log10(sig0tab[iIso][:arrayLength])
                y_sigC = sigCtab[iIso][:arrayLength, ig]
                y_sigL = sigLtab[iIso][:arrayLength, ig]

                # NumPy approach
                #temp_sigC = np.interp(log10sig0, x, y_sigC)
                #temp_sigL = np.interp(log10sig0, x, y_sigL)

                # SciPy approach
                #interp_sigC = sp.interpolate.interp1d(x, y_sigC)
                #interp_sigL = sp.interpolate.interp1d(x, y_sigL)
                #temp_sigC = interp_sigC(log10sig0)
                #temp_sigL = interp_sigL(log10sig0)

                # Numba approach
                temp_sigC = interp1d_numba(x, y_sigC, log10sig0)
                temp_sigL = interp1d_numba(x, y_sigL, log10sig0)

                if np.isnan(temp_sigC) or np.isnan(temp_sigL):
                    # If any of the interpolated values is NaN, replace the entire row with non-zero elements
                    nonzero_indices = np.nonzero(sigCtab[iIso][:arrayLength, ig])
                    sigC[iIso, ig] = sigCtab[iIso][nonzero_indices[0][0], ig]
                    sigL[iIso, ig] = sigLtab[iIso][nonzero_indices[0][0], ig]
                else:
                    sigC[iIso, ig] = temp_sigC
                    sigL[iIso, ig] = temp_sigL

    sigS = np.zeros((3, len(H2OB["isoName"]), H2OB["ng"], H2OB["ng"]))
    for i in range(3):
        for j in range(len(H2OB["isoName"])):
            sigS[i][j] = boosted_interpSigS(i, H2OB["isoName"][j], H2OB['temp'], H2OB['sig0'][j, :])
            #sigS[i][j] = interpSigS(i, H2OB["isoName"][j], H2OB['temp'], H2OB['sig0'][j, :])
            
    print('Done.')

    # Macroscopic cross section [1/cm] is microscopic cross section for the 
    # molecule [barn] times the number density [number of molecules/(barn*cm)]
    H2OB['SigC'] = np.transpose(sigC) @ aDen
    H2OB['SigL'] = np.transpose(sigL) @ aDen

    H2OB_SigS = np.zeros((3, H2OB['ng'], H2OB['ng']))
    for j in range(3):
        H2OB_SigS[j] =  np.transpose(
                        sigS[j][0] * aDen[0] + sigS[j][1] * aDen[1] + \
                        sigS[j][2] * aDen[2] + sigS[j][3] * aDen[3] )

    H2OB['Sig2'] =  np.transpose(
                    hdf5_H01.get('sig2_G').get('sig2') * aDen[0] + \
                    hdf5_O16.get('sig2_G').get('sig2') * aDen[1] + \
                    hdf5_B10.get('sig2_G').get('sig2') * aDen[2] + \
                    hdf5_B11.get('sig2_G').get('sig2') * aDen[3] )

    H2OB['SigT'] = H2OB['SigC'] + H2OB['SigL'] + np.sum(H2OB_SigS[0], axis=0) + np.sum(H2OB['Sig2'], axis=0)

    # Add SigS matrices to dictionary
    H2OB['SigS'] = {}
    for i in range(3):
        H2OB['SigS'][f'SigS[{i}]'] = H2OB_SigS[i]

    H2OB['SigP'] = [0.0]

    # Make a file name which includes the isotope name and the temperature
    if H2OB['temp'] < 1000:
        matName = f"macro421_H2OB__{round(H2OB['temp'])}K"  # name of the file with a temperature index
    else:
        matName = f"macro421_H2OB_{round(H2OB['temp'])}K"  # name of the file with a temperature index

    # Change the units of number density from 1/(barn*cm) to 1/cm2
    H2OB['numDen'] = H2OB['numDen']*1e24

    #------------------------------------------------------------------
    # Round the data according to the initial accuracy of the ENDF data
    nRows = H2OB["numDen"].shape[0]
    for i in range(nRows):
        H2OB["numDen"][i] = "%12.5e" % H2OB["numDen"][i]

    num_rows, num_cols = H2OB["sig0"].shape
    for i in range(num_rows):
        for j in range(num_cols):
            H2OB["sig0"][i, j] = "%13.6e" % H2OB["sig0"][i, j]

    nRows = H2OB["SigC"].shape[0]
    for i in range(nRows):
        H2OB["SigC"][i] = "%13.6e" % H2OB["SigC"][i]

    #nRows = H2OB["SigL"].shape[0]
    #for i in range(nRows):
    #    H2OB["SigL"][i] = "%13.6e" % H2OB["SigL"][i]

    #nRows = H2OB["SigF"].shape[0]
    #for i in range(nRows):
    #    H2OB["SigF"][i] = "%13.6e" % H2OB["SigF"][i]

    #num_rows, num_cols = H2OB["SigP"].shape
    #for i in range(num_rows):
    #    for j in range(num_cols):
    #        H2OB["SigP"][i, j] = "%13.6e" % H2OB["SigP"][i, j]    

    num_rows, num_cols = H2OB["Sig2"].shape
    for i in range(num_rows):
        for j in range(num_cols):
            H2OB["Sig2"][i, j] = "%13.6e" % H2OB["Sig2"][i, j]   

    nRows = H2OB["SigT"].shape[0]
    for i in range(nRows):
        H2OB["SigT"][i] = "%13.6e" % H2OB["SigT"][i]

    num_rows, num_cols = H2OB["SigS"]["SigS[0]"].shape
    for k in range(len(H2OB["SigS"].keys())):
        for i in range(num_rows):
            for j in range(num_cols):
                H2OB["SigS"][f"SigS[{k}]"][i, j] = "%13.6e" % H2OB["SigS"][f"SigS[{k}]"][i, j]  

    # nRows = H2OB["chi"].shape[0]
    # for i in range(nRows):
    #     H2OB["chi"][i] = "%13.6e" % H2OB["chi"][i]
    #------------------------------------------------------------------

    # Finally create the file with macroscopic cross sections
    writeMacroXS(H2OB, matName)

    # Close HDF5 files
    hdf5_H01.close()
    hdf5_O16.close()
    hdf5_B10.close()
    hdf5_B11.close()

if __name__ == '__main__':
    main()
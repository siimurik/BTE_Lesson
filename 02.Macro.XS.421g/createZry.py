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

def matpro():
    with h5py.File("..//00.Lib/matprop_UO2_zircaloy.h5", "w") as hdf:
        # UO2: theoretical density (kg/m3) MATPRO(2003) p. 2-56
        fuel_rho = 10980
        # UO2: specific heat (J/kg-K)
        #fuel_cp = lambda T: 162.3 + 0.3038 * T - 2.391e-4 * T**2 + 6.404e-8 * T**3
        fuel_cp = "162.3 + 0.3038 * T - 2.391e-4 * T**2 + 6.404e-8 * T**3"
        
        # UO2 thermal conductivity (W/m-K) MATPRO(2003)
        #fuel_k = lambda T, Bu, por: (1 / (0.0452 + 0.000246 * T + 0.00187 * Bu + 0.038 * (1 - 0.9 * np.exp(-0.04 * Bu))) + 3.5e9 * np.exp(-16360 / T) / T**2) * 1.0789 * (1 - por) / (1 + por / 2)
        fuel_k = "(1 / (0.0452 + 0.000246 * T + 0.00187 * Bu + 0.038 * (1 - 0.9 * np.exp(-0.04 * Bu))) + 3.5e9 * np.exp(-16360 / T) / T**2) * 1.0789 * (1 - por) / (1 + por / 2)"
        
        # UO2: thermal expansion (m/m) MATPRO(2003)
        #fuel_thExp = lambda T: (T / 1000 - 0.3 + 4 * np.exp(-5000 / T)) / 100
        fuel_thExp = "(T / 1000 - 0.3 + 4 * np.exp(-5000 / T)) / 100"

        # UO2: Young's modulus (MPa) MATPRO(2003) p. 2-58
        #fuel_E = lambda T, por: 2.334e5 * (1 - 2.752 * por) * (1 - 1.0915e-4 * T)
        fuel_E = "2.334e5 * (1 - 2.752 * por) * (1 - 1.0915e-4 * T)"
        
        # UO2: Poisson ratio (-) MATPRO(2003) p. 2-68
        fuel_nu = 0.316

        # UO2: swelling rate (1/s) MATPRO(2003)
        # fuel_swelRate = lambda dFdt, F, T: 2.5e-29 * dFdt + (T < 2800) * (8.8e-56 * dFdt * (2800 - T)**11.73 * np.exp(-0.0162 * (2800 - T)) * np.exp(-8e-27 * F))
        fuel_swelRate = "2.5e-29 * dFdt + (T < 2800) * (8.8e-56 * dFdt * (2800 - T)**11.73 * np.exp(-0.0162 * (2800 - T)) * np.exp(-8e-27 * F))"
        
        # UO2: thermal creep rate (1/s) a simplified correlation for sensitivity study
        #fuel_creepRate = lambda sig, T: 5e5 * sig * np.exp(-4e5 / (8.314 * T))
        fuel_creepRate = "5e5 * sig * np.exp(-4e5 / (8.314 * T))"

        #############################################################################

        # He: thermal conductivity (W/m-K)
        #gap_kHe = lambda T: 2.639e-3 * T**0.7085
        gap_kHe = "2.639e-3 * T**0.7085"

        # Xe: thermal conductivity (W/m-K)
        #gap_kXe = lambda T: 4.351e-5 * T**0.8616
        gap_kXe = "4.351e-5 * T**0.8616"

        # Kr: thermal conductivity (W/m-K)
        #gap_kKr = lambda T: 8.247e-5 * T**0.8363
        gap_kKr = "8.247e-5 * T**0.8363"

        # auxiliary function for gas mixture gas conductivity calculation (-) MATPRO
        #psi = lambda k1, k2, M1, M2: (1 + np.sqrt(np.sqrt(M1 / M2) * k1 / k2))**2 / np.sqrt(8 * (1 + M1 / M2))
        psi = "(1 + np.sqrt(np.sqrt(M1 / M2) * k1 / k2))**2 / np.sqrt(8 * (1 + M1 / M2))"

        # psi = lambda k1, k2, M1, M2: (1 + np.sqrt(np.sqrt(M1 / M2) * k1 / k2))**2 * (1 + 2.41 * (M1 - M2) * (M1 - 0.142 * M2) / np.sqrt(M1 + M2)) / np.sqrt(8 * (1 + M1 / M2)); ??
        # gas mixture gas conductivity (-) MATPRO
        #gap_kGasMixFun = lambda k, x, M: k[0]*x[0]/(psi(k[0], k[0], M[0], M[0]) * x[0] + psi(k[0], k[1], M[0], M[1]) * x[1] + psi(k[0], k[2], M[0], M[2]) * x[2]) + \
        #                                 k[1]*x[1]/(psi(k[1], k[0], M[1], M[0]) * x[0] + psi(k[1], k[1], M[1], M[1]) * x[1] + psi(k[1], k[2], M[1], M[2]) * x[2]) + \
        #                                 k[2]*x[2]/(psi(k[2], k[0], M[2], M[0]) * x[0] + psi(k[2], k[1], M[2], M[1]) * x[1] + psi(k[2], k[2], M[2], M[2]) * x[2])
        gap_kGasMixFun = "  k[0]*x[0]/(psi(k[0], k[0], M[0], M[0]) * x[0] + psi(k[0], k[1], M[0], M[1]) * x[1] + psi(k[0], k[2], M[0], M[2]) * x[2]) + \
                            k[1]*x[1]/(psi(k[1], k[0], M[1], M[0]) * x[0] + psi(k[1], k[1], M[1], M[1]) * x[1] + psi(k[1], k[2], M[1], M[2]) * x[2]) + \
                            k[2]*x[2]/(psi(k[2], k[0], M[2], M[0]) * x[0] + psi(k[2], k[1], M[2], M[1]) * x[1] + psi(k[2], k[2], M[2], M[2]) * x[2])"

        # Zry: density (kg/m3)
        clad_rho = 6600
        # Zry: specific heat (J/kg-K)
        #clad_cp = lambda T: 252.54 + 0.11474 * T
        clad_cp = "252.54 + 0.11474 * T"

        # Zry thermal conductivity (W/m-K)
        #clad_k = lambda T: 7.51 + 2.09e-2 * T - 1.45e-5 * T**2 + 7.67e-9 * T**3
        clad_k = "7.51 + 2.09e-2 * T - 1.45e-5 * T**2 + 7.67e-9 * T**3"

        # Zry: thermal expansion (-) PNNL(2010) p.3-16
        #clad_thExp = lambda T: -2.373e-5 + (T - 273.15) * 6.721e-6
        clad_thExp = "-2.373e-5 + (T - 273.15) * 6.721e-6"

        # Zry: Young's modulus (MPa) PNNL(2010) p. 3-20 (cold work assumed zero)
        #clad_E = lambda T: 1.088e5 - 54.75 * T
        clad_E = "1.088e5 - 54.75 * T"

        # Zry: Poisson ratio (-) MATPRO(2003) p. 4-242
        clad_nu = 0.3

        # Zry: thermal creep rate (1/s) a simplified correlation for sensitivity study
        #clad_creepRate = lambda sig, T: 1e5 * sig * np.exp(-2e5 / (8.314 * T))
        clad_creepRate = "1e5 * sig * np.exp(-2e5 / (8.314 * T))"

        # Zry: strength coefficient MATPRO
        #clad_K = lambda T:  (T < 743)   * (2.257e9 + T * (-5.644e6 + T * (7.525e3 - T * 4.33167))) + \
        #                    (T >= 743)  * (T < 1090) * (2.522488e6 * np.exp(2.8500027e6 / T**2)) + \
        #                    (T >= 1090) * (T < 1255) * (184.1376039e6 - 1.4345448e5 * T) + \
        #                    (T >= 1255) * (4.330e7 + T * (-6.685e4 + T * (37.579 - T * 7.33e-3)))
        clad_K = "(T < 743)   * (2.257e9 + T * (-5.644e6 + T * (7.525e3 - T * 4.33167))) + \
                (T >= 743)  * (T < 1090) * (2.522488e6 * np.exp(2.8500027e6 / T**2)) + \
                (T >= 1090) * (T < 1255) * (184.1376039e6 - 1.4345448e5 * T) + \
                (T >= 1255) * (4.330e7 + T * (-6.685e4 + T * (37.579 - T * 7.33e-3)))"
        # Zry: strain rate sensitivity exponent MATPRO
        #clad_m = lambda T:  (T <= 730) * 0.02 + \
        #                    (T > 730) * (T <= 900) * (20.63172161 - 0.07704552983 * T + 9.504843067e-05 * T**2 - 3.860960716e-08 * T**3) + \
        #                    (T > 900) * (-6.47e-02 + T * 2.203e-04)
        clad_m = "  (T <= 730) * 0.02 + \
                    (T > 730) * (T <= 900) * (20.63172161 - 0.07704552983 * T + 9.504843067e-05 * T**2 - 3.860960716e-08 * T**3) + \
                    (T > 900) * (-6.47e-02 + T * 2.203e-04)"
        # Zry: strain hardening exponent MATPRO
        #clad_n = lambda T:  (T < 1099.0772) * (-9.490e-2 + T * (1.165e-3 + T * (-1.992e-6 + T * 9.588e-10))) + \
        #                    (T >= 1099.0772) * (T < 1600) * (-0.22655119 + 2.5e-4 * T) + \
        #                    (T >= 1600) * 0.17344880
        clad_n = "  (T < 1099.0772) * (-9.490e-2 + T * (1.165e-3 + T * (-1.992e-6 + T * 9.588e-10))) + \
                    (T >= 1099.0772) * (T < 1600) * (-0.22655119 + 2.5e-4 * T) + \
                    (T >= 1600) * 0.17344880"
        # Zry: burst stress  MATPRO(2003) p.4-187
        #clad_sigB = lambda T: 10**(8.42 + T * (2.78e-3 + T * (-4.87e-6 + T * 1.49e-9))) / 1e6
        clad_sigB = "10**(8.42 + T * (2.78e-3 + T * (-4.87e-6 + T * 1.49e-9))) / 1e6"

        #############################################################################

        fuel_group = hdf.create_group("fuel")
        fuel_group.create_dataset("rho", data=fuel_rho)
        fuel_group.attrs['cp'] = fuel_cp
        fuel_group.attrs['k'] = fuel_k
        fuel_group.attrs['thExp'] = fuel_thExp
        fuel_group.attrs['E'] = fuel_E
        fuel_group.create_dataset("nu", data=fuel_nu)
        fuel_group.attrs['swelRate'] = fuel_swelRate
        fuel_group.attrs['psi'] = psi
        fuel_group.attrs['creepRate'] = fuel_creepRate

        gap_group = hdf.create_group("gap")
        gap_group.attrs['kHe'] = gap_kHe
        gap_group.attrs['kXe'] = gap_kXe
        gap_group.attrs['kKr'] = gap_kKr
        gap_group.attrs['kGasMixFun'] = gap_kGasMixFun

        clad_group = hdf.create_group("clad")
        clad_group.create_dataset("rho", data=clad_rho)
        clad_group.attrs['cp'] = clad_cp
        clad_group.attrs['k'] = clad_k
        clad_group.attrs['thExp'] = clad_thExp
        clad_group.attrs['E'] = clad_E
        clad_group.create_dataset("nu", data=clad_nu)
        clad_group.attrs['creepRate'] = clad_creepRate
        clad_group.attrs['K'] = clad_K
        clad_group.attrs['m'] = clad_m
        clad_group.attrs['n'] = clad_n
        clad_group.attrs['sigB'] = clad_sigB

def read_matpro(input_keyword):
    # Define the mapping of input keyword to group name
    group_mapping = {
                    "clad": "clad",
                    "fuel": "fuel",
                    "gap": "gap"
                    }
    data = {}
    with h5py.File("..//00.Lib/matprop_UO2_zircaloy.h5", "r") as file:
        # Get the group name based on the input keyword
        group_name = group_mapping.get(input_keyword)
        if group_name is not None:
            # Get the group
                group = file[group_name]
                attrs = list(group.attrs.keys())
                datasets = list(group.keys())

                for attr_name in attrs:
                    data[attr_name] = group.attrs.get(attr_name)

                for dataset_name in datasets:
                    data[dataset_name] = np.array(group.get(dataset_name))

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
                # NumPy Method
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
            f'Zry temperature:    {s_struct["temp"]:.1f} K',
            f'Zry density:        {s_struct["den"]:.5f} g/cm3',
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

    The function reads the MICROsopic group cross sections in the HDF5
    format and calculates from them the MACROscopic cross sections for
    natural mixture of zirconium isotopes which is the basis of zircalloy --
    fuel cladding material of the pressurized water reactor.
    =========================================================================
    """
    # input and initialize the geometry of the PWR unit cell (the function is in '..\00.Lib')
    lib_path = os.path.join('..', '00.Lib')
    file_path_PWR = os.path.join(lib_path, 'initPWR_like.h5')

    if not os.path.exists(file_path_PWR):
        # File doesn't exist, call initPWR_like() function
        initPWR_like()
    # Read in the necessary data struct. Options: {fr, g, th}
    g = readPWR_like("g")

    # number of energy groups
    Zry = {}
    Zry["ng"] = 421

    # Zircaloy is composed of pure Zirconium (~1% of tin neglected). There are
    # four stable isotopes of zirconium: Zr090, Zr091, Zr092, Zr094 and one
    # very long-lived: Zr096 with the following molar fractions:
    molFrZr = np.array([0.5145, 0.1122, 0.1715, 0.1738, 0.0280])

    # Path to microscopic cross section data:
    micro_XS_path = '../01.Micro.XS.421g'                                # INPUT

    # Call the functions for Zr isotopes and store the data in the structures.
    # As an example it is done below for 600K, change when other temperatures needed:
    hdf5_ZR90 = h5py.File(micro_XS_path + '/micro_ZR090__600K.h5', 'r')  # INPUT
    print(f"File 'micro_ZR090__600K.h5' has been read in.")
    hdf5_ZR91 = h5py.File(micro_XS_path + '/micro_ZR091__600K.h5', 'r')  # INPUT
    print(f"File 'micro_ZR091__600K.h5' has been read in.")
    hdf5_ZR92 = h5py.File(micro_XS_path + '/micro_ZR092__600K.h5', 'r')  # INPUT
    print(f"File 'micro_ZR092__600K.h5' has been read in.")
    hdf5_ZR94 = h5py.File(micro_XS_path + '/micro_ZR094__600K.h5', 'r')  # INPUT
    print(f"File 'micro_ZR094__600K.h5' has been read in.")
    hdf5_ZR96 = h5py.File(micro_XS_path + '/micro_ZR096__600K.h5', 'r')  # INPUT
    print(f"File 'micro_ZR096__600K.h5' has been read in.")

    Zry["temp"] = 600                                                       # INPUT
    Zry["eg"] = np.array(hdf5_ZR90.get('en_G').get('eg'))
    Zry['ng'] = 421

    # Mass of one "average" Zr atom in atomic unit mass [a.u.m.]
    Zry["aw"] = hdf5_ZR90.attrs.get('aw')*molFrZr[0] + \
                hdf5_ZR91.attrs.get('aw')*molFrZr[1] + \
                hdf5_ZR92.attrs.get('aw')*molFrZr[2] + \
                hdf5_ZR94.attrs.get('aw')*molFrZr[3] + \
                hdf5_ZR96.attrs.get('aw')*molFrZr[4]

    # The function matpro() sets up the data for some material properties 
    # of uranium dioxide, inert gases and zircaloy
    file_path_matpro = os.path.join(lib_path, 'matprop_UO2_zircaloy.h5')
    if not os.path.exists(file_path_matpro):
        matpro()
    # read_matpro() returns the material properties of Zry in structure "clad"
    clad = read_matpro("clad")

    # Zircaloy density
    Zry["den"] = clad["rho"]*1e-3   # [g/cm3]
    rho = Zry["den"]*1.0e-24        # [g/(barn*cm)]
    rho = rho / 1.660538e-24        # [(a.u.m.)/(barn*cm)]
    rho = rho / Zry["aw"]           # [number of Zr atoms/(barn*cm)]

    # The names of isotopes
    Zry["isoName"] = ['ZR90', 'ZR91', 'ZR92', 'ZR94', 'ZR96']

    # The number densities of isotopes
    Zry["numDen"] = np.array([molFrZr[0] * rho, molFrZr[1] * rho, molFrZr[2] * rho, 
                            molFrZr[3] * rho, molFrZr[4] * rho])

    # Prepare for sigma-zero iterations:
    sigTtab = prepareIntoND(
                prep2D(hdf5_ZR90.get('sigT_G')), 
                prep2D(hdf5_ZR91.get('sigT_G')), 
                prep2D(hdf5_ZR92.get('sigT_G')), 
                prep2D(hdf5_ZR94.get('sigT_G')), 
                prep2D(hdf5_ZR96.get('sigT_G'))
            )

    sig0_sizes = []
    for i in range(len(Zry["isoName"])):
        sig0_sizes.append(len(np.array(eval(f'hdf5_{Zry["isoName"][i]}').get('sig0_G').get('sig0'))))

    sig0tab = np.zeros((len(Zry["isoName"]), max(sig0_sizes)))
    for i, size in enumerate(sig0_sizes):
        #print("i =", i, "size =", size)
        sig0tab[i, :size] = np.array(eval(f'hdf5_{Zry["isoName"][i]}').get('sig0_G').get('sig0'))

    aDen = Zry['numDen']

    # SigEscape -- escape cross section, for simple convex objects (such as
    # plates, spheres, or cylinders) is given by S/(4V), where V and S are the
    # volume and surface area of the object, respectively
    SigEscape = 1 / (2 * g["clad"]["rOut"] * 100)

    print('Sigma-zero iterations.')
    Zry["sig0"] = sigmaZeros(sigTtab, sig0tab, aDen, SigEscape)
    print('Done.')

    print('Interpolation of microscopic cross sections for the found sigma-zeros. ')
    sigCtab = prepareIntoND(
            prep2D(hdf5_ZR90.get('sigC_G')), 
            prep2D(hdf5_ZR91.get('sigC_G')), 
            prep2D(hdf5_ZR92.get('sigC_G')), 
            prep2D(hdf5_ZR94.get('sigC_G')),
            prep2D(hdf5_ZR96.get('sigC_G'))
            )
#-------------------------------------------------------------------------------------------
    sigL_sizes = []
    for i in range(len(Zry["isoName"])):
        sigL_data = np.array(eval(f'hdf5_{Zry["isoName"][i]}').get('sigL_G').get('sigL'))
        sigL_sizes.append(len(sigL_data))

    maxL_size = max(sigL_sizes)
    sigLtab = np.zeros((len(Zry["isoName"]), maxL_size, Zry["ng"]))
    col_start = 0
    for i, size in enumerate(sigL_sizes):
        sigL_data = np.array(eval(f'hdf5_{Zry["isoName"][i]}').get('sigL_G').get('sigL'))
        sigLtab[i, :size, col_start:col_start+421] = sigL_data.reshape(size, 421)
#-------------------------------------------------------------------------------------------

    sigC, sigL = np.zeros((len(aDen), Zry['ng'])), np.zeros((len(aDen), Zry['ng']))

    for ig in range(Zry['ng']):
        # Number of isotopes in the mixture
        nIso = len(aDen)
        # Loop over isotopes
        for iIso in range(nIso):
            # Find cross sections for the found sigma-zeros
            if np.count_nonzero(sig0tab[iIso]) == 1:
                sigC[iIso, ig] = sigCtab[iIso][0, ig]
                sigL[iIso, ig] = sigLtab[iIso][0, ig]
            else:
                log10sig0 = min(10, max(0, np.log10(Zry['sig0'][iIso, ig])))
                arrayLength = len(sig0tab[iIso][np.nonzero(sig0tab[iIso])])
                x = np.log10(sig0tab[iIso][:arrayLength])
                y_sigC = sigCtab[iIso][:arrayLength, ig]
                y_sigL = sigLtab[iIso][:arrayLength, ig]

                # NumPy approach
                #temp_sigC = np.interp(log10sig0, x, y_sigC)
                #temp_sigL = np.interp(log10sig0, x, y_sigL)

                # SciPy approach
                interp_sigC = sp.interpolate.interp1d(x, y_sigC)
                interp_sigL = sp.interpolate.interp1d(x, y_sigL)
                temp_sigC = interp_sigC(log10sig0)
                temp_sigL = interp_sigL(log10sig0)

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

    sigS = np.zeros((3, len(Zry["isoName"]), Zry["ng"], Zry["ng"]))
    for i in range(3):
        for j in range(len(Zry["isoName"])):
            sigS[i][j] = boosted_interpSigS(i, Zry["isoName"][j], Zry['temp'], Zry['sig0'][j, :])
            #sigS[i][j] = interpSigS(i, Zry["isoName"][j], Zry['temp'], Zry['sig0'][j, :])
            
    print('Done.')

    #  Macroscopic cross section [1/cm] is microscopic cross section for the 
    # "average" atom [barn] times the number density [number of atoms/(barn*cm)]
    Zry['SigC'] = np.transpose(sigC) @ aDen
    Zry['SigL'] = np.transpose(sigL) @ aDen

    Zry_SigS = np.zeros((3, Zry['ng'], Zry['ng']))
    for j in range(3):
        Zry_SigS[j] =  np.transpose(
                        sigS[j][0] * aDen[0] + sigS[j][1] * aDen[1] + sigS[j][2] * aDen[2] + \
                        sigS[j][3] * aDen[3] + sigS[j][4] * aDen[4])

    Zry['Sig2'] =  np.transpose(
                    hdf5_ZR90.get('sig2_G').get('sig2') * aDen[0] + \
                    hdf5_ZR91.get('sig2_G').get('sig2') * aDen[1] + \
                    hdf5_ZR92.get('sig2_G').get('sig2') * aDen[2] + \
                    hdf5_ZR94.get('sig2_G').get('sig2') * aDen[3] + \
                    hdf5_ZR96.get('sig2_G').get('sig2') * aDen[4] )

    Zry['SigT'] = Zry['SigC'] + Zry['SigL'] + np.sum(Zry_SigS[0], axis=0) + np.sum(Zry['Sig2'], axis=0)

    # Add SigS matrices to dictionary
    Zry['SigS'] = {}
    for i in range(3):
        Zry['SigS'][f'SigS[{i}]'] = Zry_SigS[i]

    # Number of fissile isotopes
    Zry["nFis"] = 0

    Zry['SigP'] = np.array([0.0])

    # Make a file name which includes the isotope name and the temperature
    if Zry['temp'] < 1000:
        matName = f"macro421_Zry__{round(Zry['temp'])}K"  # name of the file with a temperature index
    else:
        matName = f"macro421_Zry_{round(Zry['temp'])}K"  # name of the file with a temperature index

    # Change the units of number density from 1/(barn*cm) to 1/cm2
    Zry['numDen'] = Zry['numDen']*1e24

    #------------------------------------------------------------------
    # Round the data according to the initial accuracy of the ENDF data
    nRows = Zry["numDen"].shape[0]
    for i in range(nRows):
        Zry["numDen"][i] = "%12.5e" % Zry["numDen"][i]

    num_rows, num_cols = Zry["sig0"].shape
    for i in range(num_rows):
        for j in range(num_cols):
            Zry["sig0"][i, j] = "%13.6e" % Zry["sig0"][i, j]

    nRows = Zry["SigC"].shape[0]
    for i in range(nRows):
        Zry["SigC"][i] = "%13.6e" % Zry["SigC"][i]

    #nRows = Zry["SigL"].shape[0]
    #for i in range(nRows):
    #    Zry["SigL"][i] = "%13.6e" % Zry["SigL"][i]

    #nRows = Zry["SigF"].shape[0]
    #for i in range(nRows):
    #    Zry["SigF"][i] = "%13.6e" % Zry["SigF"][i]

    #num_rows, num_cols = Zry["SigP"].shape
    #for i in range(num_rows):
    #    for j in range(num_cols):
    #        Zry["SigP"][i, j] = "%13.6e" % Zry["SigP"][i, j]    

    num_rows, num_cols = Zry["Sig2"].shape
    for i in range(num_rows):
        for j in range(num_cols):
            Zry["Sig2"][i, j] = "%13.6e" % Zry["Sig2"][i, j]   

    nRows = Zry["SigT"].shape[0]
    for i in range(nRows):
        Zry["SigT"][i] = "%13.6e" % Zry["SigT"][i]

    num_rows, num_cols = Zry["SigS"]["SigS[0]"].shape
    for k in range(len(Zry["SigS"].keys())):
        for i in range(num_rows):
            for j in range(num_cols):
                Zry["SigS"][f"SigS[{k}]"][i, j] = "%13.6e" % Zry["SigS"][f"SigS[{k}]"][i, j]  

    # nRows = Zry["chi"].shape[0]
    # for i in range(nRows):
    #     Zry["chi"][i] = "%13.6e" % Zry["chi"][i]
    #------------------------------------------------------------------

    # Finally create the file with macroscopic cross sections
    writeMacroXS(Zry, matName)

    # Close HDF5 files
    hdf5_ZR90.close()
    hdf5_ZR91.close()
    hdf5_ZR92.close()
    hdf5_ZR94.close()
    hdf5_ZR96.close()

if __name__ == '__main__':
    main()
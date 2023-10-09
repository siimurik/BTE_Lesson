import os
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse import find

"""
=======================================================
*The purpose of this code is not 100% understood, but 
this is what I have gathered.*

This function reads n 'words' from a specific row of a 
given matrix and returns them in a vector, along with 
the row number where the last word was read.
-------------------------------------------------------
INPUT:
n - integer; the total number of words that need to be 
    extracted from the m matrix (whatever that means);
iRow - integer; the index of the starting row of 
       the "m" matrix;
m - 2D NumPy array; essentially a matrix containing 
    the data to be extracted.
-------------------------------------------------------
NOTE: Why "m" is a NumPy array, not a Pandas DataFrame?
-------------------------------------------------------
NumPy is usually much more forgiving if you are trying 
to access an element in an array. Essentially:

        a[x,y] = a[x][y].

For a Pandas DF, you would have to use .iloc[]
which gets very bothersome, very quicly. Also,
NumPy is just faster. Simple as that.
-------------------------------------------------------
OUTPUT:
a - 1D NumPy array;
iRowNew - integer; updated row number
=======================================================
"""
def extractNwords(n, iRow, m):
    k = 0   # counter for filling vector a
    iRowNew = iRow
    #a = np.empty((1,n), dtype=np.float64)
    a = []
    for ii in range(int(n/6)):  # read lines with 6 words each
        for jj in range(6):
            #a[0][k] = m[iRowNew][jj]
            a.append(m[iRowNew][jj])
            k += 1
        iRowNew += 1

    if (n - int(n/6)*6) == 0:   # check if there's a partial line with less than 6 words
        iRowNew -= 1    # if yes, stay on the same row for next call to extractNwords()

    for jj in range((n-int(n/6)*6)):  # read the last line with less than 6 words
        #a[0][k] = m[iRowNew][jj]
        a.append(m[iRowNew][jj])
        k += 1

    a = np.array(a)
    return a, iRowNew

"""
==========================================================
In an ENDF type of file, MF 3 refers to the section that 
provides the cross-sections for outgoing neutrons produced 
by a particular reaction. Specifically, MF 3 contains the 
angular distributions and energy distributions of the 
outgoing neutrons, which are necessary for determining 
how the neutrons will interact with matter in subsequent 
calculations. The data in MF 3 is typically provided in 
tabular form as a function of incident neutron energy and 
scattering angle. This function extracts this data from
the preprocessed CSV file.
---
This code defines a function called "extract_mf3" that 
takes three inputs: mt, ntt, and m, and returns an output 
called sig. The purpose of this function is to search the 
matrix m for cross sections sig from file mf=3 for a 
specified reaction mt and temperature ntt, and return 
sig(ng,nSig0), where ng is the number of energy groups 
and nSig0 is the number of sigma-zeros.
----------------------------------------------------------
INPUT:
mt  - integer; reaction
ntt - integer; index for temperature value
m - 2D NumPy array;
----------------------------------------------------------
OUTPUT:
sig(nSig0, enGroup)- 2D NumPy array; where "enGroup" is 
the number of energy groups and "nSig0" is the the number 
of sigma-zeros.
==========================================================
"""
def extract_mf3(mt, ntt, m):
    nRow = m.shape[0]  # number of rows
    # ntemp starts from -1 to ensure 
    # Python indexing
    nTemp = -1  # number of temperatures
    iRowFound = 0
    #sig = []
    for iRow in range(nRow):
        #if m.iloc[iRow, 7] == 3 and m.iloc[iRow, 8] == mt and m.iloc[iRow, 9] == 1:
        if m[iRow, 7] == 3 and m[iRow, 8] == mt and m[iRow, 9] == 1:
            # find the row with mf=3 and required mt
            nTemp += 1  # number of temperatures
            if nTemp == ntt: 
                iRowFound = iRow + 1
                break
    if iRowFound > 0:  # there is mf=3 and required mt for this isotope
        nSig0 = int(m[iRowFound-1, 3])  # number of sigma-zeros
        nLgn = int(m[iRowFound-1, 2])  # number of Legendre components
        iRow = iRowFound + 1
        enGroup = int(m[2, 2])
        sig = np.zeros((nSig0, enGroup))
        while m[iRow, 7] == 3 and m[iRow, 8] == mt:
            ig = int(m[iRow-1, 5])
            a, iRowNew = extractNwords(nSig0 * nLgn * 2, iRow, m)
            sig[0:nSig0, ig-1] = a[nSig0*nLgn:(nSig0*nLgn+nSig0)] # Benefits of using NumPy: a[x,y] = a[x][y]
            #sig.append(a[nSig0*nLgn:(nSig0*nLgn+nSig0)])
            iRow = iRowNew + 2                                       
    else:
        #sig.append(0)
        sig = np.zeros(1)
    # Make list into NumPy array
    #sig = np.array(sig)
    return sig

"""
==============================================================
MF 6 stands for "Cross Sections" in an ENDF (Evaluated Nuclear
Data File) type of file. This section contains the evaluated 
cross-section data, such as the neutron-induced reaction cross 
sections, as a function of incident neutron energy. The cross-
section data may be provided in tabular or in resonance format. 
Additionally, the section may also contain the angular distri-
butions, energy-angle distributions, or other related data. 
This information is essential for the calculation of the neut-
ron transport in nuclear reactors and other applications.
---
This code extracts cross sections from a matrix m that stores 
nuclear data for different isotopes, reactions, and 
temperatures. The matrix is in the ENDF format. The function 
specifically extracts cross sections for a given reaction mt 
and temperature ntt from the subsection of m that has mf=6 
(multiplicity representation) data
---
In the context of nuclear physics, the "multiplicity 
representation" refers to a way of describing the number of 
particles emitted in a particular type of nuclear reaction, 
such as a fission or an (n,xn) reaction.
--------------------------------------------------------------
MAIN TASKS:
 1. Initialize variables (iRow, nTemp, ifrom, ito, sig)
 2. Loop through each row of the ENDF file (until the end is 
 reached)
 3. If the current row contains the MF/MT we are looking for, 
 extract the data
 4. Store the data in a list (sig)
--------------------------------------------------------------
INPUT:
mt  - integer; reaction;
ntt - integer; index for temperature value;
m - 2D NumPy array; contains the ENDF data
--------------------------------------------------------------
OUTPUT:
ifrom - 1D NumPy array;
ito   - 1D NumPy array;
sig   - 3D NumPy array;
==============================================================
"""
def extract_mf6(mt, ntt, m):
    iRow = 0 # row number               
    nTemp = -1 # number of temperatures; "-1" for Python indexing; "0" for Matlab indexing
    # "ifrom" and "ito" will be both defined as empty lists
    # and bc depending on the CSV file, their sizes can change.
    # Later, they will be made into a NumPy arrays, bc it's 
    # fast way to do scientific computing and can be easily 
    # converted into other libraries, if need be.
    ifrom = [] # index of group 'from'
    ito = [] # index of group 'to'
    # "sig" starts out as a 1D array, but will later be 
    # reshaped into a 3D array.
    sig = []
    
    while m[iRow,6] != -1: # up to the end
        if m[iRow,7] == 6 and m[iRow,8] == mt: # find the row with mf=6 & mt
            if m[iRow,9] == 1: # this is the first line of mf=6 & mt: initialize
                nonz = 0 # number of nonzeros
                nLgn = int(m[iRow,2]) # number of Legendre components
                nSig0 = int(m[iRow,3]) # number of sigma-zeros
                iRow += 1
                nTemp += 1 # temperature index

            ng2 = int(m[iRow,2]) # number of secondary positions
            ig2lo = int(m[iRow,3]) # index to lowest nonzero group
            nw = int(m[iRow,4]) # number of words to be read
            ig = int(m[iRow,5]) # current group index
            iRow += 1     
            a, iRowNew = extractNwords(nw, iRow, m) # extract nw words in vector a
            iRow = iRowNew

            if nTemp == ntt:
                k = nLgn*nSig0 # the first nLgn*nSig0 words are flux -- skip.
                for iTo in range(ig2lo, ig2lo+ng2-1):
                    nonz += 1
                    ifrom.append(ig)
                    ito.append(iTo)
                    for iSig0 in range(int(nSig0)):
                        for iLgn in range(int(nLgn)):
                            k += 1
                            #print("iLgn =", iLgn, "iSig0 =", iSig0, "nonz-1 =",nonz-1, "a[k-1] =",a[k-1] )                            
                            ## NOTE: THIS WORKS, DON'T TOUCH!!!!
                            data = np.append((iLgn, iSig0, nonz-1), a[k-1])
                            sig.append(data)
                            ## NOTE: THIS WORKS, DON'T TOUCH!!!!
                        #if nLgn == 1:
                        #    for i in range(2):
                        #        data = np.append((i, iSig0, nonz-1), a[k-1])
                        #        sig.append(data)    # makes it an array of arrays
        iRow += 1
        
    if nTemp == -1:
        #sigFinal = 0
        sigFinal = np.zeros(1)
    else:
        sig   = np.stack(sig)           # makes the array of arrays into one array
        #sigShape = sig.shape[0]         # get the length of the 1D array
        z = int(sig.shape[0]/(nLgn*nSig0))  # find how long is the 3rd dimesion of the 3D array
        # sig[:, sig.shape[1]-1] - take all the elements from the last column
        # containing all the data;
        # reshape((nLgn, nSig0, z), order='F') - the 2D shape of the 3D array 
        # can be defined by the values of "nLgn" and "nSig0". "z" has been
        # calculated based on the length of the 1D array. "order" describes
        # the way data is stored in the final ("sigFinal" 3D matrix). Specifically,
        # "order='F'" signifies the Fortran-like indexing order, which Matlab uses,
        # bc in Matlab, data is stored row-wise rather than column-wise like in
        # C or Python (order = 'C'; the default option for reshape)
        sigFinal = sig[:, sig.shape[1]-1].reshape((nLgn, nSig0, z), order='F')
        if nLgn == 1:   # Not sure why this is necessary, maybe for discrete ordinate method?
            # create two rows of zeros with shape (2, nSig0, sigShape)
            rows_of_zeros = np.zeros((2, nSig0, z))
            # stack original array and rows of zeros vertically
            sigFinal = np.vstack((sigFinal, rows_of_zeros))
    
    # Convert lists to NumPy arrays for faster processing
    # and for (ifrom == 0).all() check to work
    ifrom = np.array(ifrom) 
    ito   = np.array(ito) 
    # Return the extracted data as well as the indices
    return ifrom, ito, sigFinal

"""
===================================================
Start of convertCSV2H5.
---------------------------------------------------
Essentially just converts CSV files into HDF5 files
for every temperature section in the CSV file.
===================================================
"""
def main():
    csv_directory = "CSV_files" # Specify the directory containing the CSV files
    csv_files = [file for file in os.listdir(csv_directory) if file.endswith('.CSV')] # Get a list of all CSV files in the csv_directory

    for csv_file in csv_files: # Loop over all CSV files
        csv_path = os.path.join(csv_directory, csv_file) # Get the full path of the CSV file
        nameOnly = os.path.splitext(csv_file)[0] # Find the name of the file without extension
        print(f"Import data from {nameOnly}.CSV. ", end="")
        m = np.genfromtxt(csv_path, delimiter=';') # Load CSV file into matrix m
        print("Done.")
        nRow = m.shape[0] # number of rows

        nTemp = 0 # number of temperatures
        temp = [] # vector of temperatures

        for iRow in range(nRow):
            if m[iRow,7] == 1 and m[iRow,8] == 451 and m[iRow,9] == 2:
                nTemp += 1 # number of temperatures
                temp.append(m[iRow,0]) # vector of temperatures)
        temp = np.array(temp)

        for iTemp in range(nTemp): # loop over temperatures
            if temp[iTemp] < 1000:
                isoName = f"micro_{nameOnly}__{round(temp[iTemp])}K" # name of the file with a temperature index
            else:
                isoName = f"micro_{nameOnly}_{round(temp[iTemp])}K" # name of the file with a temperature index

            h5_file_path = isoName + '.h5' # HDF5 file name (no need to specify the directory as it will be created in the same directory as the Python script)
            if not os.path.exists(h5_file_path): # If the corresponding HDF5 file does not exist
                print(f"Check if the HDF5 file for {isoName} is already available.")
                with h5py.File(h5_file_path, 'w') as hdf:
                    # Add important parameters for which the microscopic cross sections were generated
                    hdf.attrs['description'] = 'Python-based Neutron Transport Simulation'

                    # write the data to the HDF5 file
                    #hdf.create_dataset('atomic_weight', data=m[1, 1] * 1.008664916)
                    # write atomic_weight (amu) to the HDF5 file as metadata
                    hdf.attrs['aw'] = m[1, 1] * 1.008664916

                    # write group number to the HDF5 file as metadata
                    ng = int(421)
                    hdf.attrs['ng'] = ng
                    
                    # write temperature to the HDF5 file as data
                    #hdf.create_dataset('temperature', data=temp[iTemp])
                    hdf.attrs['temperature'] = temp[iTemp]

                    # extract and sigma-zeros and write into metadata
                    nSig0 = int(m[1, 3])
                    hdf.attrs['nSig0'] = nSig0

                    # extract the energy group boundaries 
                    a = extractNwords(int(1 + nSig0 + (ng+1)), 3, m)
                    

                    # write energy group boundaries to the HDF5 file as a dataset
                    eg_G = hdf.create_group("en_G")
                    eg_G.create_dataset('eg', data=a[0][int(1+nSig0) : int(2+nSig0+ng)])
                    #eg = a[0][int(1+nSig0) : int(2+nSig0+ng)]
                    #hdf.attrs['energy_group_boundaries'] = eg

                    # write sigma-zeros to the HDF5 file as a dataset
                    sig0_G = hdf.create_group("sig0_G")
                    sig0_G.create_dataset('sig0', data=a[0][1 : int(2+nSig0-1)])
                    #sig0 = a[0][1 : int(2+nSig0-1)]
                    #hdf.attrs['sigma_zeros'] = sig0
                    
                    # write the number of sigma zeros to the HDF5 file as data
                    sig0_G.create_dataset('nSig0', data=nSig0)
                    #hdf.attrs['num_sigma_0'] = nSig0

                    #============================================================================================
                    # (n,gamma)
                    # The notation (n, gamma) represents a neutron capture reaction, where a neutron (n) is
                    # captured by a target nucleus and a gamma-ray (gamma) is emitted. This reaction is also
                    # sometimes referred to as radiative capture, as the gamma-ray emission indicates the release
                    # of energy from the system. The (n, gamma) reaction is an important process in nuclear 
                    # astrophysics, as it is responsible for the creation of heavy elements in stars. It is also 
                    # an important process in nuclear engineering, as it is used in neutron detectors and in the 
                    # production of radioisotopes for medical and industrial applications.
                    # Extract mf=3 mt=102 (radiative capture cross sections)
                    #============================================================================================
                    print(f"Convert {nameOnly}.CSV to {isoName}.h5: mf=3 mt=102 radiative capture")
                    sigC = extract_mf3(102, iTemp, m)
                    nSig0C = sigC.shape[0]

                    sigC_G = hdf.create_group('sigC_G')
                    for iSig0 in range(nSig0C):
                        sigC_G.create_dataset(f"sigC({iSig0},:)", data=sigC[iSig0,0:ng])
                    if nSig0C == 1 and nSig0 > 1:
                        sigC_G.create_dataset('1:nSig0', data=sigC[0,0:ng])

                    #============================================================================================
                    # (n,alfa)
                    # The notation (n,α) refers to a type of nuclear reaction where a neutron (n) is absorbed by 
                    # a target nucleus, resulting in the emission of an alpha particle (α). This is also known as 
                    # an (n,α) reaction. The alpha particle has a charge of +2 and a mass of 4, and is therefore 
                    # a helium nucleus. (n,α) reactions are important in nuclear physics and nuclear engineering, 
                    # and they are used, for example, in the production of radioisotopes for medical and 
                    # industrial applications.
                    #============================================================================================
                    print(f"Convert {nameOnly}.CSV to {isoName}.h5: mf=3 mt=107 (n,alfa)")
                    sigL = extract_mf3(107, iTemp, m)  # Extract mf=3 mt=107 (production of an alfa particle)
                    #if sigL.size == 0:
                    sigL_G = hdf.create_group("sigL_G")
                    if (sigL == 0).all():
                        sigL = np.zeros((nSig0, ng))
                        sigL_G.create_dataset('sigL', data=sigL)
                    else:
                        nSig0L = sigL.shape[0]
                        for iSig0 in range(nSig0L):
                            sigL_G.create_dataset(f"sigL({iSig0},:)", data=sigL[iSig0,0:ng])
                        if nSig0L == 1 and nSig0 > 1:
                            sigL = np.tile(sigL, (nSig0, 1))
                            sigL_G.create_dataset('sigL', data=sigL)

                    #============================================================================================
                    # (n,2n)
                    # The notation (n,2n) represents a type of nuclear reaction that occurs when a neutron (n) 
                    # collides with a nucleus and causes it to emit two neutrons. This type of reaction is a type 
                    # of neutron-induced reaction, and it is a common way for neutrons to be absorbed by a nucleus. 
                    # In other words, a nucleus that absorbs a neutron in this type of reaction will usually emit 
                    # two neutrons. The (n,2n) reaction can occur with a variety of target nuclei, and it is an 
                    # important reaction in nuclear engineering and in the study of nuclear reactions.
                    #============================================================================================
                    print(f"Convert {nameOnly}.CSV to {isoName}.h5: mf=6 mt=16 (n,2n) reaction")
                    ifrom2, ito2, sig2 = extract_mf6(16, iTemp, m)  # Extract mf=6 mt=16 ((n,2n) matrix)
                    sig2_G = hdf.create_group('sig2_G')
                    #print("%% (n,2n) matrix for 1 Legendre component")
                    #if ifrom2[0] == 0:
                    if (ifrom2 == 0).all():
                        isn2n = 0
                        sig2 = np.zeros((ng, ng))
                        sig2_G.create_dataset('sig2', data=sig2)
                    else:
                        isn2n = 1
                        sig2_G.create_dataset('ifrom2', data=ifrom2)
                        sig2_G.create_dataset('ito2', data=ito2)
                        sig2_sparse = sparse.coo_matrix((sig2[0, 0, :], (ifrom2-1, ito2-1)), shape=(ng, ng))
                        sig2_new = sig2_sparse.toarray()
                        sig2_G.create_dataset('sig2', data=sig2_new)
                    
                    #============================================================================================
                    # (n,n')
                    # The notation (n,n') represents a neutron inelastic scattering reaction, where a neutron is 
                    # scattered by a nucleus, resulting in the emission of a different type of particle or gamma 
                    # ray. In this notation, the "n" inside the parentheses represents the incident neutron, and 
                    # the "n'" outside the parentheses represents the neutron that is scattered by the nucleus. 
                    # This reaction is often used to study the properties of the target nucleus, such as its 
                    # energy levels and excitation states.
                    #============================================================================================
                    igThresh = 95  # last group of thermal energy (e = 4 eV)
                    print(f'Convert {nameOnly}.CSV to {isoName}.h5: mf=6 mt=2 elastic scattering')
                    ifromE, itoE, sigE = extract_mf6(2, iTemp, m)  # Extract mf=6 mt=2 (elastic scattering matrix)
                    sigE_G = hdf.create_group('sigE_G')
                    nLgn = sigE.shape[0]-1  # nLgn = 6
                    sigS = [[np.zeros((ng, ng)) for _ in range(nSig0)] for _ in range(nLgn+1)]
                    for jLgn in range(nLgn + 1):    # 6 + 1
                        for iSig0 in range(nSig0):
                            for ii in range(len(ifromE)):
                                if ifromE[ii] <= igThresh:
                                    sigE[jLgn, iSig0, ii] = 0         # 6(+1) 5(+1)
                            sigS[jLgn][iSig0] = sparse.coo_matrix((sigE[jLgn, iSig0, :]+1e-30, (ifromE-1, itoE-1)), shape=(ng, ng))
                            #print(sigS[0][0].toarray())    # To see the first cell
                            # Also just in case you are interesed in 
                            # seeing the dimensions of sigS:
                            #for jLgn in range(nLgn + 1):
                            #    for iSig0 in range(nSig0):
                            #        print(f"Shape of sigS[{jLgn}][{iSig0}]: {sigS[jLgn][iSig0].shape}")
                    sigE_G.create_dataset('sigE', data=sigE)
                    
                    for ii in range(51, 92):
                        ifromI, itoI, sigI = extract_mf6(ii, iTemp, m) # Extract mf=6 mt=51 ... 91 (inelastic scattering matrix)
                        if len(ifromI) > 0 and ifromI[0] > 0:
                            print(f'Convert {nameOnly}.CSV to {isoName}.h5: mf=6 mt={ii:2d} inelastic scattering')
                            nLgn = sigI.shape[0]-1
                            for jLgn in range(nLgn+1):
                                for iSig0 in range(nSig0):
                                    sigS[jLgn][iSig0] += sparse.coo_matrix((sigI[jLgn, 0]+1e-30, (ifromI-1, itoI-1)), shape=(ng, ng))

                    if isoName[0:11] == 'micro_H_001':
                        print(f'Convert {nameOnly}.CSV to {isoName}.h5: mf=6 mt=222 thermal scattering for hydrogen binded in water')
                        ifromI, itoI, sigI = extract_mf6(222, iTemp, m) # Extract mf=6 mt=222 thermal scattering for hydrogen binded in water
                    else:
                        print(f'Convert {nameOnly}.CSV to {isoName}.h5: mf=6 mt=221 free gas thermal scattering')
                        ifromI, itoI, sigI = extract_mf6(221, iTemp, m) # Extract mf=6 mt=221 free gas thermal scattering

                    nLgn = sigI.shape[0] - 1
                    for jLgn in range(nLgn + 1):
                        for iSig0 in range(nSig0):
                            sigS[jLgn][iSig0] += sparse.coo_matrix((sigI[jLgn, 0]+1e-30, (ifromI-1, itoI-1)), shape=(ng, ng))
                            #sigS[jLgn][iSig0] = sigS[jLgn][iSig0] + sparse.csr_matrix((sigI[jLgn, 0]+1e-30)*np.ones(len(ifromI)), (ifromI, itoI), shape=(int(ng), int(ng)))
                    
                    sigS_G = hdf.create_group("sigS_G")
                    for jLgn in range(3):
                        for iSig0 in range(nSig0):
                            ifromS_, itoS_, sigS_ = find(sigS[jLgn][iSig0])
                            sigS_sparse = sparse.coo_matrix((sigS_, (ifromS_, itoS_)), shape=(ng, ng))
                            sigS_new = sigS_sparse.toarray()
                            sigS_G.create_dataset(f"sigS({jLgn},{iSig0})", data=sigS_new)
                    sigS_G.create_dataset("ifromS", data=ifromS_)
                    sigS_G.create_dataset("itoS", data=itoS_)

                    #============================================================================================
                    # (n,fis)
                    # The notation (n,fission) or (n,fis) refers to a nuclear reaction where a neutron (n) is 
                    # absorbed by a target nucleus and the resulting compound nucleus undergoes fission, 
                    # releasing a varying number of neutrons and other nuclear fragments (fission products). 
                    # The reaction is often written as:
                    #   n + target nucleus → compound nucleus → fission products + neutrons + energy
                    # The number of neutrons released in a fission event can vary depending on the target nucleus 
                    # and the incident neutron energy. This reaction is important in nuclear reactors where the 
                    # released neutrons can initiate a chain reaction that generates energy.
                    #============================================================================================
                    print(f"Convert {nameOnly}.CSV to {isoName}.h5: mf=3 mt=18 (fission cross sections)")
                    sigF = extract_mf3(18, iTemp, m)  # Extract mf=3 mt=18 (fission cross sections)
                    sigF_G = hdf.create_group("sigF_G")
                    nubar_G = hdf.create_group("nubar_G")
                    chi_G = hdf.create_group("chi_G")
                    if np.all(sigF == 0):
                        # fission cross sections (b)
                        sigF = np.zeros((nSig0, ng))
                        sigF_G.attrs['fissile'] = 0
                        sigF_G.create_dataset('sigF', data=sigF)
                        sigF_G.attrs['comment'] = '(n,fis)'
                        #=====================================================================
                        # nubar
                        # nubar is an important parameter in nuclear reactor physics and plays 
                        # a crucial role in modelling the neutron transport. nubar is the 
                        # average number of neutrons produced per fission event. It is an 
                        # important quantity because it determines the multiplication factor 
                        # (k-effective) of a nuclear reactor, which is a measure of whether 
                        # the reactor is critical (k-effective = 1) or supercritical 
                        # (k-effective > 1) or subcritical (k-effective < 1).
                        #=====================================================================
                        nubar = np.zeros((nSig0,ng))
                        nubar_G.create_dataset('nubar', data = nubar)
                        # fission spectrum
                        chi = np.zeros((nSig0,ng))
                        chi_G.create_dataset('chi', data=chi)
                    else:
                        print(f"Convert {nameOnly}.CSV to {isoName}.h5: mf=3 mt=18 fission")
                        sigF_G.attrs['fissile'] = 1
                        # fission cross sections (b) for {nSig0F} sigma-zero(s)
                        nSig0F = sigF.shape[0]
                        for iSig0 in range(nSig0F):
                            sigF_G.create_dataset(f"sigF({iSig0},:)", data=sigF[iSig0, 0:ng])

                        nubar = extract_mf3(452, iTemp, m)  # Extract mf=3 mt=452 (total nubar)
                        print(f"Convert {nameOnly}.CSV to {isoName}.h5: mf=3 mt=452 total nubar")

                        nSig0nu = nubar.shape[0]
                        for iSig0 in range(nSig0nu):
                            nubar_G.create_dataset(f"nubar({iSig0},:)", data=nubar[iSig0, 0:ng])

                        #============================================================================
                        # chi
                        # The chi function in the nuclear Boltzmann transport equation represents the 
                        # distribution of neutrons produced by fission events. It describes the 
                        # probability that a fission event will produce a neutron with a certain 
                        # energy. Specifically, the chi function is defined as the product of two 
                        # terms: the prompt fission neutron spectrum, which describes the energy 
                        # distribution of neutrons emitted within a few microseconds of a fission 
                        # event, and the delayed neutron precursor distribution, which describes the 
                        # probability that a neutron precursor will decay and emit a neutron with a 
                        # certain energy. The chi function is a crucial input parameter for nuclear 
                        # reactor simulations, as it influences the behaviour of the neutron 
                        # population and the energy production in the reactor.
                        #============================================================================
                        print(f"Convert {nameOnly}.CSV to {isoName}.h5: mf=6 mt=18 fission spectrum")
                        iRow = 0
                        while not (m[iRow, 7] == 6 and m[iRow, 8] == 18): # find fission spectrum
                            iRow += 1
                        iRow += 1
                        ig2lo = int(m[iRow, 3]) # index to lowest nonzero group
                        nw = int(m[iRow, 4]) # number of words to be read
                        iRow += 1
                        a = extractNwords(nw, iRow, m)[0] # read nw words in vector a
                        chi = np.zeros(ng)
                        for iii in range(ig2lo-1):
                            chi[iii] = 0.0
                        for iii in range(nw):
                            chi[iii+ig2lo-1] = a[iii]
                        # fission spectrum
                        chi_G.create_dataset('chi', data=chi/np.sum(chi))

                    # Calculate total cross sections (note that mf=3 mt=1 does not include upscatters).
                    sigT_G = hdf.create_group("sigT_G")
                    sigT = np.empty((nSig0,ng))
                    for iSig0 in range(nSig0):
                        # Compute the sum of the iSig0th row of sigS (using sparse.toarray() and np.sum())
                        sigS_sum = np.sum(sigS[0][iSig0].toarray(), axis=1)
                        # Add sigC(iSig0,:), sigF(iSig0,:), sigL(iSig0,:), and the sum to sigT(iSig0,:)
                        #sigT[iSig0,:] = sigC[iSig0] + sigF[iSig0] + sigL[iSig0] + sigS_sum
                        sigT[iSig0,:] = sigC[iSig0,:] + sigF[iSig0,:] + sigL[iSig0,:] + sigS_sum
                        if isn2n:
                            sigT[iSig0,:] += np.sum(sig2[0,0,:])
                        sigT_G.create_dataset(f"sigT({iSig0},:)", data=sigT[iSig0,:])
                

                    print(f"Data for {isoName} saved to HDF5 file.")
                    # File is automatically closed when the "with" block is exited
            
            #print(f'End of conversion for {nameOnly}.CSV to {isoName}.h5.')
            else:
                print(f"HDF5 file for {isoName} already exists.")
            # end of the if condition that checks if the file already exists
        # end of the loop over temperatures

if __name__ == '__main__':
    main()

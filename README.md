# BTE
Files for the master thesis on the topic named:
## Development of a simplified Monte Carlo Neutron Transport routine in Python

List of all the Python packages used:
* [re](https://docs.python.org/3/library/re.html)
* [os](https://docs.python.org/3/library/os.html)
* [time](https://docs.python.org/3/library/time.html)
* [h5py](https://docs.h5py.org/en/stable/)
* [tqdm](https://github.com/tqdm/tqdm)
* [NumPy](https://numpy.org/)
* [Numba](https://numba.readthedocs.io/en/stable/)
* [SciPy](https://scipy.org/)
* [Requests](https://pypi.org/project/requests/)
* [ZipFile](https://docs.python.org/3/library/zipfile.html)
* [pyXSteam](https://github.com/drunsinn/pyXSteam)
* [Cython](https://cython.readthedocs.io/en/latest/)
* [Matplotlib](https://matplotlib.org/)

---
### Step 1. Download nuclear data from IAEA site.

Folder: **01.Micro.XS.421g**

Code to complete task: **downloadGXS.py**

Download the GENDF files for the required isotopes from the open-access 
[IAEA website](https://www-nds.iaea.org/ads/adsgendf.html)

The isotopes used for the PWR-like unit cell calculations:

* B_010.GXS
* B_011.GXS
* H_001.GXS
* O_016.GXS
* U_235.GXS
* U_238.GXS
* ZR090.GXS
* ZR091.GXS
* ZR092.GXS
* ZR094.GXS
* ZR096.GXS

---
### Step 2. Convert data from GXS to CSV format.

Folder: **01.Micro.XS.421g**

Code to complete task: **convertGXS2CSV.py**

Run the code **convertGXS2CSV.py**. The function scans the folder */GXS_files*
that was generated by the file **downloadGXS.py** for the files with the 
extension .GXS which are microscopic cross sections in 421 energy % group 
structure in the GENDF format and convert them to the CSV ("Comma Separated 
Values") file format readable by Excel and Python. Note that semicolons are 
used in the script instead of commas, therefore, check that your regional 
settings are set to use semicolons instead of commas as the list separator symbol.

---
### Step 3. Convert data from CSV to HDF5 format.

Folder: **01.Micro.XS.421g**

Code to complete task (slower: pure Python): **convertCSV2H5.py**

Code to complete task (faster: Numba opt.): **boostedCSV2H5.py**

The default option to complete this task is to run the code **boostedCSV2H5.py**. 
However, doing this using pure Python can be extremely timeconsuming, which is why
a second code was written that can be 15x times faster than the original, named
**boostedCSV2H5.py**. What makes this code much more faster is a package named
[Numba](https://numba.readthedocs.io/en/stable/). It is a just-in-time compiler for 
Python that works best on code that uses NumPy arrays and functions, and loops.
The function scans the folder */CSV_files* for the files with the extension .CSV 
which are microscopic cross sections in 421 energy group structure in the CSV 
("Comma-Separated Value") format, obtained from the GENDF format, and convert 
them to the HDF5 format.

For every temperature available in the GENDF file, separate HDF5 dataset is created.

The HDF5 format file for an isotope includes:
- atomic weight (amu);
- number of energy groups (=421);
- energy group boundaries (eV);
- set of background cross sections, sigma-zeros (b)
- temperature (K);
- radiative capture cross sections (b) for each sigma-zero;
- (n,alfa) reaction cross sections (b) for each sigma-zero;
- (n,2n) matrix (b) for the first Legendre component;
- scattering matrix (b) for the three Legendre components and each sigma-zero;
- fission cross sections (b) for each sigma-zero;
- nubar (-);
- fission spectrum (-);
- total cross sections (b) for each sigma-zero (b);

Note that (n,3n), etc. reactions are NOT included in the current version.

---
### Step 4. Calculate macroscopic cross sections for water solution of boric acid.

Folder: **02.Macro.XS.421g**

Code to complete task: **createH2OB.py**

Run the code **createH2OB.py**. The function reads the MICROscopic group cross 
sections in the HDF5 format from folder 01.Micro.XS.421g and calculates 
from them the MACROscopic cross sections for water solution of boric acid 
which is similar to the coolant of the pressurized water reactor.

**P.S.** Ensure the [pyXSteam](https://github.com/drunsinn/pyXSteam) is installed.

---
### Step 5. Calculate macroscopic cross sections for natural zirconium.

Folder: **02.Macro.XS.421g**

Code to complete task: **createZry.py**

Run the code **createZry.py**. The function reads the MICROsopic group cross 
sections in the HDF5 format from folder 01.Micro.XS.421g and calculates 
from them the MACROscopic cross sections for natural mixture of zirconium 
isotopes which is the basis of zircalloy -- fuel cladding material of the 
pressurized water reactor.

---
### Step 6. Calculate macroscopic cross sections for uranium dioxide.

Folder: **02.Macro.XS.421g**

Code to complete task: **createUO2_03.py**

Run the code **createUO2_03.py**. The function reads the MICROscopic group cross 
sections in the HDF5 format and calculates from them the MACROscopic cross 
sections for uranium dioxide which is the fuel material of the pressurized 
water reactor.

---
### Step 7. Run the Monte-Carlo method solver

Folder: **06.Monte.Carlo**

Code to complete task (pure Python): **MonteCarloPWR.py**

Code to complete task (Cython opt.): **setup.py** -> **mc_Cython.py**

Code to complete task (Numba opt.): **numba_mc.py**

The code calculates the neutron transport in a 2D (x,y) unit cell
similar to the unit cell of the pressurized water reactor using the Monte
Carlo method. The **MonteCarloPWR.py** is written in pure Python, but 
**mc_Cython.py**, which uses [Cython](https://cython.readthedocs.io/en/latest/), 
can offer faster execution times. The goal of using Cython is to speed up the 
Monte-Carlo process by translating timeconsuming functions into optimized 
C/C++ code and compiling them as Python extension modules. 

For this, Cython assumes the existence of the *Python.h* header file. To 
ensure it is downloaded, use the following command.

For `apt`:
```
sudo apt-get install python-dev   # for python2.x installs
sudo apt-get install python3-dev  # for python3.x installs
```

For `dnf`:
```
sudo dnf install python2-devel  # for python2.x installs
sudo dnf install python3-devel  # for python3.x installs
```
To run the Cython version, run the **setup.py** file to set up the Python modules with the command  
```
python3 setup.py build_ext --inplace
```

After which you can run the **mc_Cython.py** like a normal Python file.

**P.S.** Unfortunately, the speedup gained using **mc_Cython.py** was not the 
most significant. On my PC there was only a 30 second speed gain.

However there was some luck with speeding up the final code with some noticable improvement. 
This was achieved again with the help of Numba. This version of the Monte-Carlo simulation 
can be found under the name **numba_mc.py**.

Initially this method did not bear much fruit compared to the original **MonteCarloPWR.py**
code. However, using the functions that were meant for the Cython version of the code, 
adding the Numba decorator to only those math-heavy functions, a notible increase in the 
final run time was achieved. 
```
=============================================================================
 Without Optimization (Pure Python):
    $ real	4m23.505s
    $ user	4m22.852s
    $ sys	 0m1.205s

 After Cython Optimization
    $ real	3m55.076s
    $ user	3m54.296s
    $ sys	 0m1.394s
    
 With Numba Optimization:
    $ real	1m28.181s
    $ user	1m28.348s
    $ sys	 0m1.709s
=============================================================================
```

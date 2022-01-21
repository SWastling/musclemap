# musclemap

## Synopsis
Calculate B1, fat-fraction, magnetisation transfer ratio or T2 maps

## Usage

```bash
musclemap [general options] [algorithm] [algorithm options] [source files] 
```

- `source files`: The image(s) containing the source data. The types of image 
data required depends on the algorithm used (see Description section below).


## Description
Calculate B1, fat-fraction, magnetisation transfer ratio or T2 maps

Below is a list of available algorithms, the source files that they require,
and a brief description of their behaviour:
 
- b1: calculate a B1 map from double-angle spin-echo data at 60&deg; and 
120&deg;. Based on, but not identical to, equation 7 in Ropele et al. MRM 
(2005). The result is expressed as a fraction of 60&deg;. Required source files:
    - `fa60`: 60&deg; flip-angle spin-echo NIfTI image
    - `fa120`: 120&deg; flip-angle spin-echo NIfTI image
    
- ff: calculate a fat-fraction map from the magnitude and phase data 
acquired with fat and water out-of-phase, in-phase and then out-of-phase again. 
Using the 3-point Dixon method described in Glover and Schneider MRM (1991). 
Required source files and arguments:
    - `m_1`: out-of-phase fat-water magnitude NIfTI image (TE 3.45 ms at 3 T)
    - `phi_1`: out-of-phase fat-water phase NIfTI image (TE 3.45 ms at 3 T)
    - `m0`: in phase fat-water magnitude NIfTI image (TE 4.6 ms at 3 T)
    - `phi0`: in phase fat-water phase NIfTI image (TE 4.6 ms at 3 T)
    - `m1`: out-of-phase fat-water magnitude NIfTI image (TE 5.75 ms at 3 T)
    - `phi1`: out-of-phase fat-water phase NIfTI image (TE 5.75 ms at 3 T)
    - `scanner`: scanner type, chose from `ge` or `siemens`
    
- mtr: calculate magnetisation transfer ratio (MTR) map from MT on and off data. 
Required source files: 
    - `mt_on`: MT on magnitude NIfTI image
    - `mt_off`:  MT off magnitude NIfTI image
    
- mtr-b1: calculate magnetisation transfer ratio (MTR) map from MT on and off 
data with correction for B1 inhomogeneities as described in Sinclair et al. NMR
in Biomedicine 2012. Required source files: 
    - `mt_on`: MT on magnitude NIfTI image
    - `mt_off`:  MT off magnitude NIfTI image
    - `fa60`: 60&deg; flip-angle spin-echo NIfTI image
    - `fa120`: 120&deg; flip-angle spin-echo NIfTI image
    - `ref`: reference NIfTI image used for masking and resampling typically  
    Dixon out-of-phase fat-water magnitude image (TE 3.45 ms at 3 T)

- t2: calculate T2 map from double-echo spin-echo data. By default the maps are
registered to a reference image. Required source files and arguments:
    - `e1`: First echo NIfTI image
    - `e2`:  Second echo NIfTI image
    - `te1`: echo time of first echo (ms)
    - `te2`: echo time of second echo (ms)
    - `ref`: reference NIfTI image used for registration typically Dixon 
    out-of-phase fat-water magnitude image (TE 3.45 ms at 3 T)

The version of `musclemap`, as stored by `git`, and of `FSL` and the `nibabel` 
library are verified at runtime

## Options

### Standard Options
- `-h`: display help message, and quit
- `-o`: output folder for results
- `-r`: register images to first source file using `flirt`
- `-m`: filename of mask image 
- `-c`: crop images `xmin` `xsize` `ymin` `ysize` `zmin` `zsize` 
(indexing starts with 0 not 1, inputting -1 for a size will set it to 
the full image extent for that dimension) using `fslroi`
- `-k`: save intermediate images     
- `-v`: view map(s) in `fsleyes`
- `-quiet`: don't display information messages or progress status
- `--version`: display version and exit

### ff  algorithm
- `-s`: separate images in x-direction during phase unwrapping. This is
recommended for lower limbs e.g. thighs or calves
- `-nb`: calculate noise-bias corrected fat-fraction maps - see Liu et al. MRM 
2007  

### mtr-b1  algorithm
- `-b1pcf`: population level B1 correction factor. This is the gradient of a 
straight line fit of B1 to MTR (default=0.0085)

## Software Requirements

- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) (version 6.0.3)

## Installing
1. Create a directory to store the package e.g.:

    ```bash
    mkdir musclemap
    ```

2. Create a new virtual environment in which to install `musclemap`:

    ```bash
    python3 -m venv musclemap-env
    ```
   
3. Activate the virtual environment:

    ```bash
    source musclemap-env/bin/activate
    ```

4. Upgrade `pip` and `build`:

    ```bash
    pip install --upgrade pip
    pip install --upgrade build
    ```

5. Install using `pip`:
    ```bash
    pip install git+https://github.com/SWastling/musclemap.git
    ```

## License
See [MIT license](./LICENSE)

## Authors and Acknowledgements
Dr Stephen Wastling 
([stephen.wastling@nhs.net](mailto:stephen.wastling@nhs.net)) based on a 
collection of scripts written in Python 2 by Chris Sinclair. This initial 
release was tested against the previous implementations of the code to ensure 
it produced identical results. 

## References
1. Ropele et al. _Assessment and Correction of B1 Induced Errors in 
Magnetization Transfer Ratio Measurements_ Magnetic Resonance in Medicine 
53:134 -140 (2005)

2. Glover G.H. and Schneider E. _Three-Point Dixon Technique for True Water / 
Fat Decomposition with B0 Inhomogeneity Correction_ MRM 18, 371-383 (1991) 

3. Liu C-Y., McKenzie C.A. , Yu H., Brittain J.H., Reeder S.B., _Fat 
quantification with IDEAL gradient echo imaging: Correction of bias from T1 and 
noise_  MRM 2007 Aug;58(2):354-64 

4. M. Jenkinson, C.F. Beckmann, T.E. Behrens, M.W. Woolrich, S.M. Smith. _FSL_ 
NeuroImage, 62:782-90, 2012

5. M. Jenkinson and S.M. Smith. _A global optimisation method for robust affine 
registration of brain images_ Medical Image Analysis, 5(2):143-156, 2001.

6. M. Jenkinson, P.R. Bannister, J.M. Brady, and S.M. Smith. _Improved 
optimisation for the robust and accurate linear registration and motion 
correction of brain images_ NeuroImage, 17(2):825-841, 2002. 

7. Sinclair C.D.J. et al. _Correcting radiofrequency inhomogeneity effects in 
skeletal muscle magnetisation transfer maps_ NMR Biomed. 2012; 25: 262–270 
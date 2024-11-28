# musclemap

## Synopsis
Calculate B1, fat-fraction, magnetisation transfer ratio (MTR) or T2 maps

## Usage

```bash
musclemap [general options] [algorithm] [algorithm options] [source files] 
```

- `source files`: NIfTI file(s) containing the source data. The types of image 
data required depend on the algorithm used as explained below:

## Description
Calculate B1, fat-fraction, magnetisation transfer ratio (MTR) or T2 maps. The 
list of available algorithms, the source files that they require, and a brief 
description of their behaviour is given below:

### b1
Calculate a B1 map from double-angle spin-echo data at 60&deg; and 
120&deg;. Based on, but not identical to, equation 7 in [Ropele et al. MRM 
(2005)](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.20310). The result 
is expressed as a fraction of 60&deg;. The source files required are:
- `fa60`: spin-echo image acquired with 60&deg; excitation flip-angle
- `fa120`: spin-echo image acquired with 120&deg; excitation flip-angle

### ff
Calculate a fat-fraction map from the magnitude and phase data 
acquired with fat and water out-of-phase, in-phase and then out-of-phase again. 
Using the 3-point Dixon method described in 
[Glover and Schneider MRM (1991)](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.1910180211). 
The source files and arguments required are:
- `m_1`: out-of-phase fat-water magnitude image (i.e. TE 3.45 ms at 3 T)
- `phi_1`: out-of-phase fat-water phase image (i.e. TE 3.45 ms at 3 T)
- `m0`: in phase fat-water magnitude image (i.e. TE 4.6 ms at 3 T)
- `phi0`: in phase fat-water phase image (i.e. TE 4.6 ms at 3 T)
- `m1`: out-of-phase fat-water magnitude image (i.e. TE 5.75 ms at 3 T)
- `phi1`: out-of-phase fat-water phase image (i.e. TE 5.75 ms at 3 T)
- `scanner`: scanner type, chose from `ge`, `siemens` or `philips`

> [!NOTE] 
> The `scl_slope` and `scl_inter` NIfTI header elements are explicitly 
> ignored. For GE data the input phase images should be -3142:3142, for Siemens 
> and Philips data 0:4095. Phase images are scaled to radians internally.

### mtr
Calculate magnetisation transfer ratio (MTR) map from MT on and off data. 
The source files required are:
- `mt_on`: MT on magnitude NIfTI image
- `mt_off`:  MT off magnitude NIfTI image
    
### mtr-b1
Calculate magnetisation transfer ratio (MTR) map from MT on and off 
data with correction for B1 inhomogeneities as described in [Sinclair et al. NMR
in Biomedicine 2012](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/nbm.1744). 
The source files required are:
- `mt_on`: MT on magnitude NIfTI image
- `mt_off`:  MT off magnitude NIfTI image
- `fa60`: 60&deg; flip-angle spin-echo NIfTI image
- `fa120`: 120&deg; flip-angle spin-echo NIfTI image
- `ref`: reference NIfTI image used for masking and resampling typically Dixon out-of-phase fat-water magnitude image (TE 3.45 ms at 3 T)

### t2
Calculate T2 map from double-echo spin-echo data. By default the maps are
registered to a reference image. Required source files and arguments:
- `e1`: First echo NIfTI image
- `e2`:  Second echo NIfTI image
- `te1`: echo time of first echo (ms)
- `te2`: echo time of second echo (ms)
- `ref`: reference NIfTI image used for registration typically Dixon 
out-of-phase fat-water magnitude image (TE 3.45 ms at 3 T)

## Options

### Standard Options
- `-h`: display help message, and quit
- `-o`: output folder for results
- `-r`: register images to first source file using FSL `flirt`
- `-m`: filename of mask image 
- `-c`: crop images `xmin` `xsize` `ymin` `ysize` `zmin` `zsize` 
(indexing starts with 0 not 1, inputting -1 for a size will set it to 
the full image extent for that dimension) using `fslroi`
- `-k`: save intermediate images     
- `-v`: view map(s) in `fsleyes`
- `-quiet`: don't display information messages or progress status
- `--version`: display version and exit
- `--any-version`: don't abort if version checks fail 

### ff  algorithm
- `-s`: separate images in x-direction during phase unwrapping. This is
recommended for lower limbs e.g. thighs or calves
- `-nb`: calculate noise-bias corrected fat-fraction maps - see [Liu et al. MRM 
2007](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21301) for details  

### mtr-b1  algorithm
- `-b1pcf`: population level B1 correction factor. This is the gradient of a 
straight line fit of B1 to MTR (default=0.0085)

## Software Requirements

- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) (version 6.0.3 or 6.0.4)

> [!NOTE] 
> The version of `FSL` is verified at runtime.
## Installing
1. Create a new virtual environment in which to install `musclemap`:

    ```bash
    uv venv musclemap-venv
    ```
   
2. Activate the virtual environment:

    ```bash
    source musclemap-venv/bin/activate
    ```

4. Install using `uv pip`:
    ```bash
    uv pip install git+https://github.com/SWastling/musclemap.git
    ```
   
> [!TIP]
> You can also run `musclemap` without installing it using 
>[uvx](https://docs.astral.sh/uv/guides/tools/) i.e. with the command 
>`uvx --from  git+https://github.com/SWastling/musclemap.git musclemap`

## License
See [MIT license](./LICENSE)

## Authors and Acknowledgements
Written by [Stephen Wastling](mailto:stephen.wastling@nhs.net) based on a 
collection of Python 2 scripts by Chris Sinclair.  

import nibabel as nib
import numpy as np

import preproc


def calc_t2(e1, e2, te1, te2):
    """
    Calculate T2 and S0 map from double-echo spin-echo images

    :param e1: echo 1 image
    :type e1: np.ndarray
    :param e2: echo 2 image
    :type e2: np.ndarray
    :param te1: echo time of echo 1 in ms
    :type te1: float
    :param te2: echo time of echo 2 in ms
    :type te2: float
    :return: s0map, t2map
    """

    # Set any negative pixels to 0
    e1[np.where(e1 < 0.0)] = 0.0
    e2[np.where(e2 < 0.0)] = 0.0

    # Calculate the T2 map image
    with np.errstate(divide='ignore', invalid='ignore'):
        t2map = (te2 - te1) / (np.log(e1) - np.log(e2))
        t2map = np.nan_to_num(t2map, copy=False, posinf=200.0, neginf=0.0)

    # Set limits same as Chris Sinclair's original code
    t2map[np.where(t2map > 200.0)] = 200.0
    t2map[np.where(t2map < 0.0)] = 0.0

    # Calculate the S0 map
    with np.errstate(divide='ignore', invalid='ignore'):
        s0map = e1 * np.exp(te1 / t2map)
        s0map = np.nan_to_num(s0map, copy=False, posinf=5000.0, neginf=0.0)

    s0map[np.where(s0map > 5000)] = 5000.0
    s0map[np.where(s0map < 0)] = 0.0

    return s0map, t2map


def process_t2(fp_dict, te1, te2, reg_fp, out_dir, to_delete, fsldir,
               quiet=True):

    """
    Calculate T2 map from double-echo spin-echo data

    :param fp_dict: filenames of double-echo spin-echo NIfTI files
    :type fp_dict: dict
    :param te1: Echo time of first image in ms
    :type te1: float
    :param te2: Echo time of first image in ms
    :type te2: float
    :param reg_fp: Filepath of image to register T2 data to e.g. out-of-phase
    Dixon
    :type reg_fp: pathlib.Path
    :param out_dir: directory to store T2 map NIfTI file in
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :type to_delete: list
    :param fsldir: full path to FSL directory
    :type fsldir: pathlib.Path
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: t2_fp, to_delete: T2 map filepath and intermediate files to delete
    :rtype: tuple
    """

    if not quiet:
        print('** registering 1st and 2nd echo data to %s with flirt'
              % reg_fp)

    fp_dict, to_delete = preproc.register_t2(fp_dict, reg_fp, out_dir,
                                             to_delete, fsldir, quiet)

    if not quiet:
        print('** loading NIfTI image data')
    e1 = nib.load(str(fp_dict['e1_fp'])).get_fdata()
    e2 = nib.load(str(fp_dict['e2_fp'])).get_fdata()
    affine_out = nib.load(str(fp_dict['e1_fp'])).header.get_best_affine()

    if not quiet:
        print('** calculating T2 map')
    s0, t2map = calc_t2(e1, e2, te1, te2)

    if not quiet:
        print('** saving NIfTI files')
    s0_fp = out_dir / 's0.nii.gz'
    s0_obj = nib.nifti1.Nifti1Image(s0, affine_out)
    s0_obj.to_filename(str(s0_fp))

    t2_fp = out_dir / 't2.nii.gz'
    t2map_obj = nib.nifti1.Nifti1Image(t2map, affine_out)
    t2map_obj.to_filename(str(t2_fp))

    return [t2_fp, s0_fp], to_delete
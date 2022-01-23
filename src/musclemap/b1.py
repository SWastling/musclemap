import nibabel as nib
import numpy as np


def calc_b1(fa60, fa120):
    """
    Calculate flip angle as a fraction of 60 degrees

    :param fa60: image intensity acquired at nominal flip angle of 60 degrees
    :type fa60: np.ndarray
    :param fa120: image intensity acquired at nominal flip angle of 120 degrees
    :type fa120: np.ndarray
    :return: flip angle as a fraction of 60 degrees
    :rtype: np.ndarray
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = (180.0 * np.arccos(fa120 / (2.0 * fa60))) / (np.pi * 60.0)
        b1 = np.nan_to_num(b1, copy=False, posinf=0.0, neginf=0.0)

    return b1


def process_b1(fp_dict, out_dir, quiet=True):
    """
    Wrapper to calculate b1 map i.e. flip angle as a fraction of 60 degrees

    :param fp_dict: dictionary of NIfTI files
    :type fp_dict: dict
    :param out_dir: directory to store b1 map NIfTI file in
    :type out_dir: pathlib.Path
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: b1_fp: filepath of b1 map NIfTI
    :rtype: pathlib.Path
    """

    if not quiet:
        print("** loading NIfTI image data")
    fa60 = nib.load(str(fp_dict["fa60_fp"])).get_fdata()
    fa120 = nib.load(str(fp_dict["fa120_fp"])).get_fdata()
    affine_out = nib.load(str(fp_dict["fa60_fp"])).header.get_best_affine()

    if not quiet:
        print("** calculating B1")
    b1 = calc_b1(fa60, fa120)

    if not quiet:
        print("** saving NIfTI file")
    b1_fp = out_dir / "b1.nii.gz"
    b1_obj = nib.nifti1.Nifti1Image(b1, affine_out)
    b1_obj.to_filename(str(b1_fp))

    return b1_fp

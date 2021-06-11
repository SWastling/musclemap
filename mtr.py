import nibabel as nib
import numpy as np


def calc_mtr(mt_on, mt_off):
    """
    Calculate MTR as a percentage

    :param mt_on: MT on image
    :type mt_on: np.ndarray
    :param mt_off: MT off image
    :type mt_off: np.ndarray
    :return: magnetisation transfer ratio as a percentage
    :rtype: np.ndarray
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        mtr = 100.0 * np.divide((mt_off - mt_on), mt_off)
        mtr = np.nan_to_num(mtr, copy=False, posinf=0.0, neginf=0.0)

    return mtr


def process_mtr(fp_dict, out_dir, quiet=True):
    """
    Wrapper to calculate MTR map

    :param fp_dict: filepaths of MT On and Off NIfTI files
    :type fp_dict: dict
    :param out_dir: directory to store MTR map NIfTI file in
    :type out_dir: pathlib.Path
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: mtr_fp: filename of MTR map NIfTI
    :rtype: pathlib.Path
    """

    if not quiet:
        print('** loading NIfTI image data')
    mt_on = nib.load(str(fp_dict['mt_on_fp'])).get_fdata()
    mt_off = nib.load(str(fp_dict['mt_off_fp'])).get_fdata()
    affine_out = nib.load(str(fp_dict['mt_on_fp'])).header.get_best_affine()

    if not quiet:
        print('** calculating MTR')
    mtr = calc_mtr(mt_on, mt_off)

    if not quiet:
        print('** saving NIfTI file')
    mtr_fp = out_dir / 'mtr.nii.gz'
    mtr_obj = nib.nifti1.Nifti1Image(mtr, affine_out)
    mtr_obj.to_filename(str(mtr_fp))

    return mtr_fp

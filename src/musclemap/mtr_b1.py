import nibabel as nib
import numpy as np

import musclemap.b1 as b1
import musclemap.mtr as mtr
import musclemap.preproc as preproc


def calc_b1pcf(mtr_masked, b1_masked, b1pcf):
    """
    Apply population wide B1 correction factor to MTR image

    :param mtr_masked: masked MTR image
    :type mtr_masked: np.ndarray
    :param b1_masked: masked B1 map
    :type b1_masked: np.ndarray
    :param b1pcf: population level B1 correction factor
    :type b1pcf: float
    :return: magnetisation transfer ratio as a percentage corrected using
    population level correction factor
    :rtype: np.ndarray
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        mtr_b1pcf = np.divide(mtr_masked,
                              (1.0 + 100.0 * b1pcf * (b1_masked - 1.0)))
        mtr_b1pcf = np.nan_to_num(mtr_b1pcf, copy=False, posinf=0.0, neginf=0.0)

    return mtr_b1pcf


def calc_b1_perc_err(b1):
    """
    Convert B1 map (fraction of 60 degrees) to B1 error as a percentage

    :param b1: B1 map as a fraction of 60 degrees
    :type b1: np.ndarray
    :return: B1 error as a percentage
    :rtype: np.ndarray
    """

    return np.multiply(100.0, np.divide((np.multiply(60.0, b1) - 60.0), 60.0))


def stderrs_from_covmat(covmat):
    """
    Convert covariance matrix to list of standard errors

    :param covmat: Covariance matrix
    :return: std_err_gradient, std_err_intercept
    """

    return np.sqrt(np.diag(covmat))


def propagate_error_div(f, a, stderr_a, b, stderr_b):

    """
    Propagate errors for function f = a / b when the standard errors in a and b
    are known

    :param f: f= a/b
    :param a: numerator
    :param stderr_a: error in numerator
    :param b: denominator
    :param stderr_b: error in denominator
    :return: standard error in f
    """

    return f * np.sqrt((stderr_a / a)**2 + (stderr_b / b)**2)


def apply_b1scf(mtr, k, b1_perc_err):
    """
    Apply subject specific B1 correction to MTR using eqn 4 from Sinclair et
    al. NMR in Biomedicine 2012; 25: 262-270

    :param mtr: MT ratio
    :param k: Gradient of linear plot of MTR against B1
    :param b1_perc_err: B1 error as percentage
    :return: mtr corrected for B1 error
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        mtr_b1scf = np.divide(mtr, (k * b1_perc_err) + 1)
        mtr_b1scf = np.nan_to_num(mtr_b1scf, copy=False, posinf=0.0, neginf=0.0)

    return mtr_b1scf


def calc_b1scf(mtr_masked, b1_masked):
    """
    Calculate subject specific B1 correction to MTR image

    As described in Sinclair et al. NMR in Biomedicine 2012; 25: 262-270

    :param mtr_masked: masked MTR image
    :type mtr_masked: np.ndarray
    :param b1_masked: masked b1 map
    :type b1_masked: np.ndarray
    :return: mtr_b1scf, fit_param_dict
    :rtype: tuple(np.ndarray, dict)
    """

    # Convert b1 map to b1 error
    b1_perc_err = calc_b1_perc_err(b1_masked)

    # Unravel arrays to vectors
    b1_perc_err_vec = np.ravel(b1_perc_err)
    mtr_masked_vec = np.ravel(mtr_masked)

    # Threshold to mtr values > 1.0 %
    mtr_thr_vec = mtr_masked_vec[mtr_masked_vec > 1.0]
    b1_thr_vec = b1_perc_err_vec[np.nonzero(mtr_masked_vec > 1.0)]

    # Fit a straight line to the data
    # y = m x +c i.e. mtr = kspec * b1_error + mtr_true
    [k_spec, mtr_true], cov = np.polyfit(b1_thr_vec, mtr_thr_vec, 1,
                                         full=False, cov=True)

    # Standard error of slope and intercept are sqrt of diagonal elements of
    # covariance matrix
    [k_spec_error, mtr_true_error] = stderrs_from_covmat(cov)

    # Eqn 5 from Sinclair et al. NMR in Biomedicine 2012; 25: 262-270
    k = k_spec / mtr_true

    k_error = propagate_error_div(k, k_spec, k_spec_error, mtr_true,
                                  mtr_true_error)

    mtr_b1scf = apply_b1scf(mtr_masked, k, b1_perc_err)

    fit_param_dict = {'mtr_true': mtr_true, 'mtr_true_error': mtr_true_error,
                      'k_spec': k_spec, 'k_spec_error': k_spec_error,
                      'k': k, 'k_error': k_error}

    return mtr_b1scf, fit_param_dict


def process_b1_correction(mtr_fp, b1_fp, bgmask_fp, out_dir, b1pcf,
                          quiet=True):

    """
    Calculate and then apply B1 correction to MTR images

    :param mtr_fp: filepath of MTR NIfTI image
    :type mtr_fp: pathlib.Path
    :param b1_fp: filepath of B1 map NIfTI image
    :type b1_fp: pathlib.Path
    :param bgmask_fp: filename of background mask NIfTI image
    :type bgmask_fp: pathlib.Path
    :param out_dir: directory to store B1 corrected MTR map NIfTI file
    :type out_dir: pathlib.Path
    :param b1pcf: population level B1 correction
    :type b1pcf: float
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: mtr_b1pcf_fp, mtr_b1scf_fp
    :rtype: pathlib.Path
    """

    if not quiet:
        print('*** loading NIfTI image data')
    mtr_obj = nib.load(str(mtr_fp))
    mtr = mtr_obj.get_fdata()
    affine_out = mtr_obj.header.get_best_affine()

    b1 = nib.load(str(b1_fp)).get_fdata()
    bgmask = nib.load(str(bgmask_fp)).get_fdata()

    if not quiet:
        print('*** masking MTR and B1 maps')
    mtr_masked = np.multiply(mtr, bgmask)
    b1_masked = np.multiply(b1, bgmask)

    if not quiet:
        print('*** calculating MTR with population-level B1 correction')

    mtr_b1pcf = calc_b1pcf(mtr_masked, b1_masked, b1pcf)

    if not quiet:
        print('*** calculating MTR with subject-specific B1 correction')

    mtr_b1scf, fit_param_dict = calc_b1scf(mtr_masked, b1_masked)

    if not quiet:
        print('*** saving subject-specifc B1 inhomogeneity correction factor')

    fit_params_fp = out_dir / 'mtrb1_fitparams.dat'
    with open(fit_params_fp, 'w') as f:
        for key, val in fit_param_dict.items():
            f.write('%s: %g\n' % (key, val))

    if not quiet:
        print('*** saving NIfTI files')
    mtr_b1pcf_fp = out_dir / 'mtr_b1pcf.nii.gz'
    mtr_b1pcf_obj = nib.nifti1.Nifti1Image(mtr_b1pcf, affine_out)
    mtr_b1pcf_obj.to_filename(str(mtr_b1pcf_fp))

    mtr_b1scf_fp = out_dir / 'mtr_b1scf.nii.gz'
    mtr_b1scf_obj = nib.nifti1.Nifti1Image(mtr_b1scf, affine_out)
    mtr_b1scf_obj.to_filename(str(mtr_b1scf_fp))

    return mtr_b1pcf_fp, mtr_b1scf_fp


def process_mtr_b1(fp_dict, fa60_fp, fa120_fp, res_ref_fp, out_dir, to_delete,
                   fsldir, b1pcf, quiet=True):

    """
    Wrapper to calculate MTR maps with B1 inhomogeneity correction
    (as in IBM/CMT and IBM4809 projects)

    :param fp_dict: filepaths of MT On and Off NIfTI files
    :type fp_dict: dict
    :param fa60_fp: spin-echo image with flip angle of 60 degrees for B1 corr
    :type fa60_fp: pathlib.Path
    :param fa120_fp: spin-echo image with flip angle of 120 degrees for B1 corr
    :type fa120_fp: pathlib.Path
    :param res_ref_fp: filepath of NIfTI file to resample the data into
    :type res_ref_fp: pathlib.Path
    :param out_dir: directory to store MTR map NIfTI file in
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :type to_delete: list
    :param fsldir: full path to FSL directory
    :type fsldir: pathlib.Path
    :param b1pcf: population level B1 correction factor
    :type b1pcf: float
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return:
    """

    if not quiet:
        print('** resampling MT on and off data to dimensions of %s'
              % res_ref_fp)

    fp_dict, to_delete = preproc.resample(fp_dict, res_ref_fp, out_dir,
                                          to_delete, fsldir, quiet)

    if not quiet:
        print('** calculating MTR map')

    mtr_fp = mtr.process_mtr(fp_dict, out_dir)

    if not quiet:
        print('** cropping double-angle spin-echo images for B1 map to central '
              'slices to match coverage of %s' % res_ref_fp)

    fa_fp_dict = {'fa60_fp': fa60_fp, 'fa120_fp': fa120_fp}
    fa_fp_dict, to_delete = preproc.crop(fa_fp_dict, [0, -1, 0, -1, 10, 20], out_dir,
                                 to_delete, fsldir, quiet)

    if not quiet:
        print('** resampling double-angle spin-echo images for B1 map to '
              'dimensions of %s' % res_ref_fp)

    fa_fp_dict, to_delete = preproc.resample(fa_fp_dict, res_ref_fp, out_dir, to_delete,
                                     fsldir, quiet)
    if not quiet:
        print('** calculating B1 map')
    b1_fp = b1.process_b1(fa_fp_dict, out_dir, quiet)

    if not quiet:
        print('** creating a mask by thresholding %s' % res_ref_fp)

    bgmask_fp, to_delete = preproc.create_mask(res_ref_fp, out_dir, to_delete, fsldir,
                                       quiet)

    if not quiet:
        print('** correcting MTR map for B1 errors')

    mtr_b1pcf_fp, mtr_b1scf_fp = process_b1_correction(mtr_fp, b1_fp, bgmask_fp,
                                                       out_dir, b1pcf, quiet)

    return [mtr_fp, mtr_b1pcf_fp, mtr_b1scf_fp], to_delete

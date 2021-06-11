import nibabel as nib
import numpy as np
import pathlib
import pytest
import subprocess as sp

import preproc
import t2
import vercheck


def perror(r_fp, t_fp):
    """
    calculate the percentage error between two nifti files; a reference and
    a test

    Based on test used in FSL Evaluation and Example Data Suite (FEEDS)

    :param r_fp: reference file
    :type r_fp: pathlib.Path
    :param t_fp: test file
    :type t_fp: pathlib.Path
    :return: percentage error of r and t
    :rtype: float
    """

    r_obj = nib.load(str(r_fp))
    # nibabel defaults to float64 so we need to explicitly check for complex
    r_type = r_obj.get_data_dtype()
    if r_type == 'complex64':
        r = r_obj.get_fdata(dtype=np.complex64)
    elif r_type == 'complex128':
        r = r_obj.get_fdata(dtype=np.complex128)
    else:
        r = r_obj.get_fdata()

    t_obj = nib.load(str(t_fp))
    t_type = t_obj.get_data_dtype()
    if t_type == 'complex64':
        t = t_obj.get_fdata(dtype=np.complex64)
    elif t_type == 'complex128':
        t = t_obj.get_fdata(dtype=np.complex128)
    else:
        t = t_obj.get_fdata()

    return 100.0 * np.sqrt(np.mean(np.square(r - t)) / np.mean(np.square(r)))


perror_path = pathlib.Path('test_data/perror/')


@pytest.mark.parametrize("ref_fp, test_fp, expected_output ",
                         [(perror_path / 'OneHundred.nii.gz',
                           perror_path / 'OneHundredOne.nii.gz',
                           1.0),
                          (perror_path / 'OneHundred.nii.gz',
                           perror_path / 'NinetyNine.nii.gz',
                           1.0),
                          (perror_path / 'OneHundred.nii.gz',
                           perror_path / 'NinetyNinePointFive.nii.gz',
                           0.5),
                          (perror_path / 'OneHundred.nii.gz',
                           perror_path / 'Zero.nii.gz',
                           100.0),
                          (perror_path / 'OneHundred.nii.gz',
                           perror_path / 'OneHundredwithGaussianNoise'
                                         'SigmaOne.nii.gz',
                           1.0003711823974208)])
def test_perror(ref_fp, test_fp, expected_output):
    assert perror(ref_fp, test_fp) == expected_output


def test_calc_t2():
    e1 = np.array([-1.0, 0.960789439152323,
                   0.020189651799466, 0.449328964117222, 6.70320046035639,
                   0.081873075307798, 0.852143788966211, 9.23116346386636])
    e2 = np.array([-1.0, 0.869358235398806,
                   0.000369786371648, 0.060810062625218, 2.46596963941606,
                   0.049658530379141, 0.571209063848815, 7.55783741455725])
    te1 = 16.0
    te2 = 56.0
    ref_s0 = np.array([0.0, 1.04081077, 0.1, 1.0, 10.0, 0.1, 1.0, 10.0])
    ref_t2 = np.array([0.0, 200.0, 10.0, 20.0, 40.0, 80.0, 100.0, 200.0])

    s0map, t2map = t2.calc_t2(e1, e2, te1, te2)
    assert np.allclose(s0map, ref_s0)
    assert np.allclose(t2map, ref_t2)


def test_process_t2(tmp_path):
    # When running on stoney with FSL 6.0.4 the co-registration step of the T2
    # map production is marginally different, but within a mask the images are
    # less than 2% different
    pthresh = 2.0

    data_dir = pathlib.Path('test_data/t2')
    input_dir = data_dir / 'input'
    output_dir = data_dir / 'output'

    e1_fp = input_dir / 't2_te16ms.nii.gz'
    e2_fp = input_dir / 't2_te56ms.nii.gz'

    fp_dict = {'e1_fp': e1_fp, 'e2_fp': e2_fp}

    reg_ref_fp = input_dir / '0007-Dixon_TE_345_cf.nii.gz'
    ref_t2_fp = output_dir / 't2_m.nii.gz'
    ref_s0_fp = output_dir / 's0_m.nii.gz'
    mask_fp = output_dir / 'mask.nii.gz'

    [t2_fp, s0_fp], to_delete = t2.process_t2(fp_dict, 16.0, 56.0, reg_ref_fp,
                                              tmp_path, [],
                                              vercheck.get_fsldir(), False)

    fp_dict = {'t2_fp': t2_fp, 's0_fp': s0_fp}

    fp_dict, to_delete = preproc.mask(fp_dict, mask_fp, tmp_path, to_delete,
                                      vercheck.get_fsldir(), False)

    assert perror(ref_t2_fp, fp_dict['t2_fp']) < pthresh
    assert perror(ref_s0_fp, fp_dict['s0_fp']) < pthresh

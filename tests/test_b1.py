import nibabel as nib
import numpy as np
import pathlib
import pytest

import musclemap.b1 as b1

THIS_DIR = pathlib.Path(__file__).resolve().parent
TEST_DATA_DIR = THIS_DIR / "test_data"


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


perror_path = TEST_DATA_DIR / 'perror'


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


@pytest.mark.parametrize('fa60, fa120, expected_output',
                         [(np.array([1.0, 2.0, 1.0, 1.0]),
                           np.array([2.0, 2.0, np.sqrt(3) * 1.0, 4.0]),
                           np.array([0.0, 1.0, 0.5, 0.0]))
                          ])
def test_b1_calc(fa60, fa120, expected_output):
    assert np.allclose(b1.calc_b1(fa60, fa120), expected_output)


def test_process_b1(tmp_path):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / 'b1'
    input_dir = data_dir / 'input'
    output_dir = data_dir / 'output'

    fa60_fp = input_dir / 'se_fa060.nii.gz'
    fa120_fp = input_dir / 'se_fa120.nii.gz'

    fp_dict = {'fa60_fp': fa60_fp, 'fa120_fp': fa120_fp}

    ref_b1_fp = output_dir / 'b1.nii.gz'

    b1_fp = b1.process_b1(fp_dict, tmp_path, False)

    assert perror(ref_b1_fp, b1_fp) < pthresh
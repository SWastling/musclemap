import pathlib

import nibabel as nib
import numpy as np
import pytest

import musclemap.mtr_b1 as mtr_b1
import musclemap.vercheck as vercheck

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
    if r_type == "complex64":
        r = r_obj.get_fdata(dtype=np.complex64)
    elif r_type == "complex128":
        r = r_obj.get_fdata(dtype=np.complex128)
    else:
        r = r_obj.get_fdata()

    t_obj = nib.load(str(t_fp))
    t_type = t_obj.get_data_dtype()
    if t_type == "complex64":
        t = t_obj.get_fdata(dtype=np.complex64)
    elif t_type == "complex128":
        t = t_obj.get_fdata(dtype=np.complex128)
    else:
        t = t_obj.get_fdata()

    return 100.0 * np.sqrt(np.mean(np.square(r - t)) / np.mean(np.square(r)))


perror_path = TEST_DATA_DIR / "perror"


@pytest.mark.parametrize(
    "ref_fp, test_fp, expected_output ",
    [
        (perror_path / "OneHundred.nii.gz", perror_path / "OneHundredOne.nii.gz", 1.0),
        (perror_path / "OneHundred.nii.gz", perror_path / "NinetyNine.nii.gz", 1.0),
        (
            perror_path / "OneHundred.nii.gz",
            perror_path / "NinetyNinePointFive.nii.gz",
            0.5,
        ),
        (perror_path / "OneHundred.nii.gz", perror_path / "Zero.nii.gz", 100.0),
        (
            perror_path / "OneHundred.nii.gz",
            perror_path / "OneHundredwithGaussianNoise" "SigmaOne.nii.gz",
            1.0003711823974208,
        ),
    ],
)
def test_perror(ref_fp, test_fp, expected_output):
    assert perror(ref_fp, test_fp) == expected_output


def test_calc_b1pcf():
    b1pcf = 0.01
    mtr_masked = np.array(
        [0.0, 0.0, 1.0, 5.0, 10.0, 50.0, 90.0, 95.0, 99.0, 100.0, 0.0, 0.0]
    )
    b1_masked = np.array([0.0, 0.0, 1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 1.0, 0.7, 0.0, 0.0])
    expected_output = np.array(
        [
            0.0,
            0.0,
            1.0,
            5.0 / 1.1,
            10.0 / 0.9,
            50.0,
            90.0 / 1.2,
            95.0 / 0.8,
            99.0,
            100.0 / 0.7,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(mtr_b1.calc_b1pcf(mtr_masked, b1_masked, b1pcf), expected_output)


def test_calc_b1_perc_err():

    b1_in = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    expected_output = np.array([-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20])
    assert np.allclose(mtr_b1.calc_b1_perc_err(b1_in), expected_output)


def test_stderrs_from_covmat():

    test_covmat = np.array([[4.0, 1.0, 0.0], [2.3, 9.0, 1.1], [0.0, 1.0, 16.0]])
    expected_output = [2.0, 3.0, 4.0]
    assert np.allclose(mtr_b1.stderrs_from_covmat(test_covmat), expected_output)


def test_propagate_error_div():
    assert mtr_b1.propagate_error_div(100.0, 100.0, 1.0, 100.0, 1.0) == np.sqrt(2.0)


def test_apply_b1scf():
    mtr = np.array([0.0, 1.0, 5.0, 10.0, 50.0, 90.0, 95.0, 100.0])
    k = 0.1
    b1_error = np.array([0.0, -10.0, -1.0, 1.0, 10.0, 0.0, 5.0, 25.0])

    expected_output = np.array(
        [0.0, 0.0, 5.0 / 0.9, 10.0 / 1.1, 50.0 / 2.0, 90.0, 95.0 / 1.5, 100.0 / 3.5]
    )

    assert np.allclose(mtr_b1.apply_b1scf(mtr, k, b1_error), expected_output)


def test_calc_b1scf():

    # MTR data generated in excel with mtr_true = 80, kspec = 0.09, with added
    # Gaussian noise with mean 0 and sd 1
    mtr = np.array(
        [
            0.626760390643661,
            0.678155193561307,
            70.5572147914649,
            72.6536444371246,
            75.4859662007084,
            74.6929656938527,
            77.2287467720388,
            80.0903889506428,
            81.804942488037,
            83.878749350007,
            85.7436033157568,
            87.4005360490854,
            89.7567004029149,
            0.849669308956752,
            -0.543099967561962,
        ]
    )

    b1 = np.array(
        [1.0, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1.1, 1.1]
    )

    expected_mtr_b1scf = np.array(
        [
            0.626760390643661,
            0.678155193561307,
            80.05838833,
            80.27516453,
            81.27317116,
            78.41544246,
            79.1063851,
            80.09038895,
            79.90827091,
            80.07737845,
            80.04392881,
            79.82203699,
            80.23461288,
            0.83970387,
            -0.53673016,
        ]
    )

    expected_fit_param_dict = {
        "mtr_true": 79.93576895014849,
        "mtr_true_error": 0.2153029286584982,
        "k_spec": 0.0948662131266116,
        "k_spec_error": 0.003404238207327976,
        "k": 0.0011867805160637713,
        "k_error": 4.27069654657691e-05,
    }

    mtr_b1scf, fit_param_dict = mtr_b1.calc_b1scf(mtr, b1)

    for key, value in expected_fit_param_dict.items():
        assert np.abs(fit_param_dict[key] - value) < 1.0e-10

    assert np.allclose(mtr_b1scf, expected_mtr_b1scf)


def test_process_b1_correction(tmp_path):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "mtr-b1"
    output_dir = data_dir / "output"

    mtr_fp = output_dir / "mtr.nii.gz"
    b1_fp = output_dir / "b1.nii.gz"
    bgmask_fp = output_dir / "bgmask.nii.gz"

    mtr_b1pcf_fp, mtr_b1scf_fp = mtr_b1.process_b1_correction(
        mtr_fp, b1_fp, bgmask_fp, tmp_path, 0.0085, False
    )

    ref_mtr_b1pcf_fp = output_dir / "mtr_b1pcf.nii.gz"
    ref_mtr_b1scf_fp = output_dir / "mtr_b1scf.nii.gz"

    assert perror(ref_mtr_b1pcf_fp, mtr_b1pcf_fp) < pthresh
    assert perror(ref_mtr_b1scf_fp, mtr_b1scf_fp) < pthresh


def test_process_mtr_b1(tmp_path):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "mtr-b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mt_on_fp = input_dir / "mt_on.nii.gz"
    mt_off_fp = input_dir / "mt_off.nii.gz"

    fp_dict = {"mt_on_fp": mt_on_fp, "mt_off_fp": mt_off_fp}

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"
    res_ref_fp = input_dir / "0006-Dixon_TE_345_th.nii.gz"
    ref_mtr_fp = output_dir / "mtr.nii.gz"
    ref_mtr_b1pcf_fp = output_dir / "mtr_b1pcf.nii.gz"
    ref_mtr_b1scf_fp = output_dir / "mtr_b1scf.nii.gz"

    [mtr_fp, mtr_b1pcf_fp, mtr_b1scf_fp], to_delete = mtr_b1.process_mtr_b1(
        fp_dict,
        fa60_fp,
        fa120_fp,
        res_ref_fp,
        tmp_path,
        [],
        vercheck.get_fsldir(),
        0.0085,
        False,
    )

    assert perror(ref_mtr_fp, mtr_fp) < pthresh
    assert perror(ref_mtr_b1pcf_fp, mtr_b1pcf_fp) < pthresh
    assert perror(ref_mtr_b1scf_fp, mtr_b1scf_fp) < pthresh

import pathlib

import nibabel as nib
import numpy as np
import pytest

import musclemap.mtr as mtr

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


@pytest.mark.parametrize(
    "test_mt_on, test_mt_off, expected_output",
    [
        (
            np.array([50.0, 0.0, 1.0, 200.0]),
            np.array([100.0, 0.0, 0.0, 100.0]),
            np.array([50.0, 0.0, 0.0, -100.0]),
        )
    ],
)
def test_calc_mtr(test_mt_on, test_mt_off, expected_output):
    assert np.allclose(mtr.calc_mtr(test_mt_on, test_mt_off), expected_output)


def test_process_mtr(tmp_path):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "mtr"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mt_on_fp = input_dir / "mt_on.nii.gz"
    mt_off_fp = input_dir / "mt_off.nii.gz"

    fp_dict = {"mt_on_fp": mt_on_fp, "mt_off_fp": mt_off_fp}

    ref_mtr_fp = output_dir / "mtr.nii.gz"

    mtr_fp = mtr.process_mtr(fp_dict, tmp_path, False)

    assert perror(ref_mtr_fp, mtr_fp) < pthresh


def test_process_mtr_quiet(tmp_path, capsys):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "mtr"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mt_on_fp = input_dir / "mt_on.nii.gz"
    mt_off_fp = input_dir / "mt_off.nii.gz"

    fp_dict = {"mt_on_fp": mt_on_fp, "mt_off_fp": mt_off_fp}

    ref_mtr_fp = output_dir / "mtr.nii.gz"

    mtr_fp = mtr.process_mtr(fp_dict, tmp_path)

    assert perror(ref_mtr_fp, mtr_fp) < pthresh

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

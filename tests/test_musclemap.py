import pathlib

import nibabel as nib
import numpy as np
import pytest
import importlib.metadata

import musclemap.preproc as preproc
import musclemap.vercheck as vercheck

THIS_DIR = pathlib.Path(__file__).resolve().parent
TEST_DATA_DIR = THIS_DIR / "test_data"
FSL_DIR = vercheck.get_fsldir()

__version__ = importlib.metadata.version("musclemap")


def perror(r_fp, t_fp):
    """
    calculate the percentage error between two nifti files; a reference and
    a test

    Based on test used in FSL Evaluation and Example Data Suite (FEEDS)

    :param r_fp: reference file
    :type r_fp: pathlib.Path
    :param t_fp: test file
    :type t_fp: pathlib.Path
    return: percentage error of r and t
    type: float
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


SCRIPT_NAME = "musclemap"
SCRIPT_USAGE = f"usage: {SCRIPT_NAME} [-h] [-o output_folder] [-r] [-m mask]"


def test_prints_help_1(script_runner):
    result = script_runner.run([SCRIPT_NAME])
    assert result.success
    assert result.stdout.startswith(SCRIPT_USAGE)


def test_prints_help_2(script_runner):
    result = script_runner.run([SCRIPT_NAME, "-h"])
    assert result.success
    assert result.stdout.startswith(SCRIPT_USAGE)


def test_prints_help_for_invalid_option(script_runner):
    result = script_runner.run([SCRIPT_NAME, "-!"])
    assert not result.success
    assert result.stderr.startswith(SCRIPT_USAGE)


def test_prints_version(script_runner):
    result = script_runner.run([SCRIPT_NAME, "--version"])
    assert result.success
    expected_version_output = SCRIPT_NAME + " " + __version__ + "\n"
    assert result.stdout == expected_version_output


def test_musclemap_b1(tmp_path, script_runner):

    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"

    ref_b1_fp = output_dir / "b1.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME, "-o", str(tmp_path), "-quiet", "b1", str(fa60_fp), str(fa120_fp)]
    )
    assert result.success

    b1_fp = tmp_path / "b1.nii.gz"

    assert perror(ref_b1_fp, b1_fp) < pthresh


def test_musclemap_b1_dir(tmp_path, script_runner, monkeypatch):

    monkeypatch.chdir(tmp_path)

    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"

    ref_b1_fp = output_dir / "b1.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME, "-v", "-quiet", "b1", str(fa60_fp), str(fa120_fp)]
    )
    assert result.success

    b1_fp = tmp_path / "b1" / "b1.nii.gz"

    assert perror(ref_b1_fp, b1_fp) < pthresh


def test_musclemap_b1_reg(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"

    ref_b1_fp = output_dir / "b1_r.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME, "-r", "-o", str(tmp_path), "b1", str(fa60_fp), str(fa120_fp)]
    )
    assert result.success

    b1_fp = tmp_path / "b1.nii.gz"

    assert perror(ref_b1_fp, b1_fp) < pthresh


def test_musclemap_b1_mask(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"
    mask_fp = input_dir / "mask.nii.gz"

    ref_b1_fp = output_dir / "b1_m.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-m",
        str(mask_fp),
        "-o",
        str(tmp_path),
        "b1",
        str(fa60_fp),
        str(fa120_fp)]
    )
    assert result.success

    b1_fp = tmp_path / "b1.nii.gz"

    assert perror(ref_b1_fp, b1_fp) < pthresh


def test_musclemap_b1_mask_quiet(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"
    mask_fp = input_dir / "mask.nii.gz"

    ref_b1_fp = output_dir / "b1_m.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-quiet",
        "-m",
        str(mask_fp),
        "-o",
        str(tmp_path),
        "b1",
        str(fa60_fp),
        str(fa120_fp)]
    )
    assert result.success

    b1_fp = tmp_path / "b1.nii.gz"

    assert perror(ref_b1_fp, b1_fp) < pthresh


def test_musclemap_b1_crop(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"

    ref_b1_fp = output_dir / "b1_c.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-c",
        "0",
        "56",
        "0",
        "-1",
        "0",
        "-1",
        "-o",
        str(tmp_path),
        "b1",
        str(fa60_fp),
        str(fa120_fp)]
    )
    assert result.success

    b1_fp = tmp_path / "b1.nii.gz"

    assert perror(ref_b1_fp, b1_fp) < pthresh


def test_musclemap_b1_crop_quiet(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"

    ref_b1_fp = output_dir / "b1_c.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-quiet",
        "-c",
        "0",
        "56",
        "0",
        "-1",
        "0",
        "-1",
        "-o",
        str(tmp_path),
        "b1",
        str(fa60_fp),
        str(fa120_fp)]
    )
    assert result.success

    b1_fp = tmp_path / "b1.nii.gz"

    assert perror(ref_b1_fp, b1_fp) < pthresh


def test_musclemap_ff_thigh(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/siemens/thigh"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mminus1_fp = input_dir / "0006-Dixon_TE_345_th.nii.gz"
    phiminus1_fp = input_dir / "0007-Dixon_TE_345_th.nii.gz"
    m0_fp = input_dir / "0008-Dixon_TE_460_th.nii.gz"
    phi0_fp = input_dir / "0009-Dixon_TE_460_th.nii.gz"
    m1_fp = input_dir / "0010-Dixon_TE_575_th.nii.gz"
    phi1_fp = input_dir / "0011-Dixon_TE_575_th.nii.gz"

    ref_ff_fp = output_dir / "fatfraction.nii.gz"
    ref_f_fp = output_dir / "fat.nii.gz"
    ref_w_fp = output_dir / "water.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-o",
        str(tmp_path),
        "-quiet",
        "ff",
        str(mminus1_fp),
        str(phiminus1_fp),
        str(m0_fp),
        str(phi0_fp),
        str(m1_fp),
        str(phi1_fp),
        "siemens",
        "-s"]
    )
    assert result.success

    ff_fp = tmp_path / "fatfraction.nii.gz"
    f_fp = tmp_path / "fat.nii.gz"
    w_fp = tmp_path / "water.nii.gz"

    assert perror(ref_ff_fp, ff_fp) < pthresh
    assert perror(ref_f_fp, f_fp) < pthresh
    assert perror(ref_w_fp, w_fp) < pthresh


def test_musclemap_ff_hand_reg(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/siemens/hand_reg"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mminus1_fp = input_dir / "0006-Dixon_TE_345_hand.nii.gz"
    phiminus1_fp = input_dir / "0007-Dixon_TE_345_hand.nii.gz"
    m0_fp = input_dir / "0008-Dixon_TE_460_hand.nii.gz"
    phi0_fp = input_dir / "0009-Dixon_TE_460_hand.nii.gz"
    m1_fp = input_dir / "0010-Dixon_TE_575_hand.nii.gz"
    phi1_fp = input_dir / "0011-Dixon_TE_575_hand.nii.gz"

    ref_ff_fp = output_dir / "fatfraction.nii.gz"
    ref_f_fp = output_dir / "fat.nii.gz"
    ref_w_fp = output_dir / "water.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-r",
        "-quiet",
        "-o",
        str(tmp_path),
        "ff",
        str(mminus1_fp),
        str(phiminus1_fp),
        str(m0_fp),
        str(phi0_fp),
        str(m1_fp),
        str(phi1_fp),
        "siemens"]
    )
    assert result.success

    ff_fp = tmp_path / "fatfraction.nii.gz"
    f_fp = tmp_path / "fat.nii.gz"
    w_fp = tmp_path / "water.nii.gz"

    assert perror(ref_ff_fp, ff_fp) < pthresh
    assert perror(ref_f_fp, f_fp) < pthresh
    assert perror(ref_w_fp, w_fp) < pthresh


def test_musclemap_ff_foot(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/siemens/foot"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mminus1_fp = input_dir / "0024-dix3d_TR23_fa05_345.nii.gz"
    phiminus1_fp = input_dir / "0025-dix3d_TR23_fa05_345.nii.gz"
    m0_fp = input_dir / "0026-dix3d_TR23_fa05_460.nii.gz"
    phi0_fp = input_dir / "0027-dix3d_TR23_fa05_460.nii.gz"
    m1_fp = input_dir / "0028-dix3d_TR23_fa05_575.nii.gz"
    phi1_fp = input_dir / "0029-dix3d_TR23_fa05_575.nii.gz"

    ref_ff_fp = output_dir / "fatfraction.nii.gz"
    ref_f_fp = output_dir / "fat.nii.gz"
    ref_w_fp = output_dir / "water.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-o",
        str(tmp_path),
        "ff",
        str(mminus1_fp),
        str(phiminus1_fp),
        str(m0_fp),
        str(phi0_fp),
        str(m1_fp),
        str(phi1_fp),
        "siemens"]
    )
    assert result.success

    ff_fp = tmp_path / "fatfraction.nii.gz"
    f_fp = tmp_path / "fat.nii.gz"
    w_fp = tmp_path / "water.nii.gz"

    assert perror(ref_ff_fp, ff_fp) < pthresh
    assert perror(ref_f_fp, f_fp) < pthresh
    assert perror(ref_w_fp, w_fp) < pthresh


def test_musclemap_ff_calf3d(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/siemens/calf3D"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mminus1_fp = input_dir / "0026-dix3d_TR23_448m_fa05_345.nii.gz"
    phiminus1_fp = input_dir / "0027-dix3d_TR23_448m_fa05_345.nii.gz"
    m0_fp = input_dir / "0028-dix3d_TR23_448m_fa05_460.nii.gz"
    phi0_fp = input_dir / "0029-dix3d_TR23_448m_fa05_460.nii.gz"
    m1_fp = input_dir / "0030-dix3d_TR23_448m_fa05_575.nii.gz"
    phi1_fp = input_dir / "0031-dix3d_TR23_448m_fa05_575.nii.gz"

    ref_ff_fp = output_dir / "fatfraction.nii.gz"
    ref_f_fp = output_dir / "fat.nii.gz"
    ref_w_fp = output_dir / "water.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-o",
        str(tmp_path),
        "ff",
        str(mminus1_fp),
        str(phiminus1_fp),
        str(m0_fp),
        str(phi0_fp),
        str(m1_fp),
        str(phi1_fp),
        "siemens",
        "-s"]
    )
    assert result.success

    ff_fp = tmp_path / "fatfraction.nii.gz"
    f_fp = tmp_path / "fat.nii.gz"
    w_fp = tmp_path / "water.nii.gz"

    assert perror(ref_ff_fp, ff_fp) < pthresh
    assert perror(ref_f_fp, f_fp) < pthresh
    assert perror(ref_w_fp, w_fp) < pthresh


def test_musclemap_ff_thigh_ge(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/ge/thigh"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mminus1_fp = input_dir / "40006-ORIG_DIXON_TE_345_TH-0000.nii.gz"
    phiminus1_fp = input_dir / "40006-ORIG_DIXON_TE_345_TH-0001.nii.gz"
    m0_fp = input_dir / "40007-ORIG_DIXON_TE_460_TH-0000.nii.gz"
    phi0_fp = input_dir / "40007-ORIG_DIXON_TE_460_TH-0001.nii.gz"
    m1_fp = input_dir / "40008-ORIG_DIXON_TE_575_TH-0000.nii.gz"
    phi1_fp = input_dir / "40008-ORIG_DIXON_TE_575_TH-0001.nii.gz"

    ref_ff_fp = output_dir / "fatfraction.nii.gz"
    ref_f_fp = output_dir / "fat.nii.gz"
    ref_w_fp = output_dir / "water.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-c",
        "0",
        "-1",
        "128",
        "256",
        "0",
        "-1",
        "-o",
        str(tmp_path),
        "ff",
        str(mminus1_fp),
        str(phiminus1_fp),
        str(m0_fp),
        str(phi0_fp),
        str(m1_fp),
        str(phi1_fp),
        "ge",
        "-s"]
    )
    assert result.success

    ff_fp = tmp_path / "fatfraction.nii.gz"
    f_fp = tmp_path / "fat.nii.gz"
    w_fp = tmp_path / "water.nii.gz"

    assert perror(ref_ff_fp, ff_fp) < pthresh
    assert perror(ref_f_fp, f_fp) < pthresh
    assert perror(ref_w_fp, w_fp) < pthresh


def test_musclemap_ff_thigh_philips_volconv(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/philips/thigh_volconv"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    # scl_slope and scl_inter are 1 and 0
    mminus1_fp = input_dir / "1801a-Dixon_TE_345_th.nii"
    phiminus1_fp = input_dir / "1801b-Dixon_TE_345_th.nii"
    m0_fp = input_dir / "2101a-Dixon_TE_460_th.nii"
    phi0_fp = input_dir / "2101b-Dixon_TE_460_th.nii"
    m1_fp = input_dir / "2401a-Dixon_TE_575_th.nii"
    phi1_fp = input_dir / "2401b-Dixon_TE_575_th.nii"

    ref_ff_fp = output_dir / "fatfraction.nii.gz"
    ref_f_fp = output_dir / "fat.nii.gz"
    ref_w_fp = output_dir / "water.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-c",
        "0",
        "-1",
        "128",
        "256",
        "0",
        "-1",
        "-o",
        str(tmp_path),
        "ff",
        str(mminus1_fp),
        str(phiminus1_fp),
        str(m0_fp),
        str(phi0_fp),
        str(m1_fp),
        str(phi1_fp),
        "philips",
        "-s"]
    )
    assert result.success

    ff_fp = tmp_path / "fatfraction.nii.gz"
    f_fp = tmp_path / "fat.nii.gz"
    w_fp = tmp_path / "water.nii.gz"

    assert perror(ref_ff_fp, ff_fp) < pthresh
    assert perror(ref_f_fp, f_fp) < pthresh
    assert perror(ref_w_fp, w_fp) < pthresh


def test_musclemap_ff_thigh_philips_dcm2niix(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/philips/thigh_dcm2niix"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    # scl_slope and scl_inter are 1 and 0
    mminus1_fp = input_dir / "1801-Dixon_TE_345_th.nii"
    phiminus1_fp = input_dir / "1801-Dixon_TE_345_th_ph.nii"
    m0_fp = input_dir / "2101-Dixon_TE_460_th.nii"
    phi0_fp = input_dir / "2101-Dixon_TE_460_th_ph.nii"
    m1_fp = input_dir / "2401-Dixon_TE_575_th.nii"
    phi1_fp = input_dir / "2401-Dixon_TE_575_th_ph.nii"

    ref_ff_fp = output_dir / "fatfraction.nii.gz"
    ref_f_fp = output_dir / "fat.nii.gz"
    ref_w_fp = output_dir / "water.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-c",
        "0",
        "-1",
        "128",
        "256",
        "0",
        "-1",
        "-o",
        str(tmp_path),
        "ff",
        str(mminus1_fp),
        str(phiminus1_fp),
        str(m0_fp),
        str(phi0_fp),
        str(m1_fp),
        str(phi1_fp),
        "philips",
        "-s"]
    )
    assert result.success

    ff_fp = tmp_path / "fatfraction.nii.gz"
    f_fp = tmp_path / "fat.nii.gz"
    w_fp = tmp_path / "water.nii.gz"

    assert perror(ref_ff_fp, ff_fp) < pthresh
    assert perror(ref_f_fp, f_fp) < pthresh
    assert perror(ref_w_fp, w_fp) < pthresh


def test_musclemap_ff_thigh_intermed(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/siemens/thigh"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mminus1_fp = input_dir / "0006-Dixon_TE_345_th.nii.gz"
    phiminus1_fp = input_dir / "0007-Dixon_TE_345_th.nii.gz"
    m0_fp = input_dir / "0008-Dixon_TE_460_th.nii.gz"
    phi0_fp = input_dir / "0009-Dixon_TE_460_th.nii.gz"
    m1_fp = input_dir / "0010-Dixon_TE_575_th.nii.gz"
    phi1_fp = input_dir / "0011-Dixon_TE_575_th.nii.gz"

    ref_ff_fp = output_dir / "fatfraction.nii.gz"
    ref_ff_nb_fp = output_dir / "fatfraction_nb.nii.gz"
    ref_f_fp = output_dir / "fat.nii.gz"
    ref_w_fp = output_dir / "water.nii.gz"
    ref_m0_left_fp = output_dir / "m0_left.nii.gz"
    ref_m0_right_fp = output_dir / "m0_right.nii.gz"
    ref_phim_left_fp = output_dir / "phim_left.nii.gz"
    ref_phim_left_uw_fp = output_dir / "phim_left_uw.nii.gz"
    ref_phim_fp = output_dir / "phim.nii.gz"
    ref_phim_right_fp = output_dir / "phim_right.nii.gz"
    ref_phim_right_uw_fp = output_dir / "phim_right_uw.nii.gz"
    ref_phim_uw_fp = output_dir / "phim_uw.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-k",
        "-o",
        str(tmp_path),
        "ff",
        str(mminus1_fp),
        str(phiminus1_fp),
        str(m0_fp),
        str(phi0_fp),
        str(m1_fp),
        str(phi1_fp),
        "siemens",
        "-s",
        "-nb"]
    )
    assert result.success

    ff_fp = tmp_path / "fatfraction.nii.gz"
    ff_nb_fp = tmp_path / "fatfraction_nb.nii.gz"
    f_fp = tmp_path / "fat.nii.gz"
    w_fp = tmp_path / "water.nii.gz"
    m0_left_fp = tmp_path / "m0_left.nii.gz"
    m0_right_fp = tmp_path / "m0_right.nii.gz"
    phim_left_fp = tmp_path / "phim_left.nii.gz"
    phim_left_uw_fp = tmp_path / "phim_left_uw.nii.gz"
    phim_fp = tmp_path / "phim.nii.gz"
    phim_right_fp = tmp_path / "phim_right.nii.gz"
    phim_right_uw_fp = tmp_path / "phim_right_uw.nii.gz"
    phim_uw_fp = tmp_path / "phim_uw.nii.gz"

    assert perror(ref_ff_fp, ff_fp) < pthresh
    assert perror(ref_ff_nb_fp, ff_nb_fp) < pthresh
    assert perror(ref_f_fp, f_fp) < pthresh
    assert perror(ref_w_fp, w_fp) < pthresh
    assert perror(ref_m0_left_fp, m0_left_fp) < pthresh
    assert perror(ref_m0_right_fp, m0_right_fp) < pthresh
    assert perror(ref_phim_left_fp, phim_left_fp) < pthresh
    assert perror(ref_phim_left_uw_fp, phim_left_uw_fp) < pthresh
    assert perror(ref_phim_fp, phim_fp) < pthresh
    assert perror(ref_phim_right_fp, phim_right_fp) < pthresh
    assert perror(ref_phim_right_uw_fp, phim_right_uw_fp) < pthresh
    assert perror(ref_phim_uw_fp, phim_uw_fp) < pthresh


def test_musclemap_ff_thigh_nb_quiet(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/siemens/thigh"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mminus1_fp = input_dir / "0006-Dixon_TE_345_th.nii.gz"
    phiminus1_fp = input_dir / "0007-Dixon_TE_345_th.nii.gz"
    m0_fp = input_dir / "0008-Dixon_TE_460_th.nii.gz"
    phi0_fp = input_dir / "0009-Dixon_TE_460_th.nii.gz"
    m1_fp = input_dir / "0010-Dixon_TE_575_th.nii.gz"
    phi1_fp = input_dir / "0011-Dixon_TE_575_th.nii.gz"

    ref_ff_nb_fp = output_dir / "fatfraction_nb.nii.gz"
    ref_f_fp = output_dir / "fat.nii.gz"
    ref_w_fp = output_dir / "water.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-quiet",
        "-o",
        str(tmp_path),
        "ff",
        str(mminus1_fp),
        str(phiminus1_fp),
        str(m0_fp),
        str(phi0_fp),
        str(m1_fp),
        str(phi1_fp),
        "siemens",
        "-s",
        "-nb"]
    )
    assert result.success

    ff_nb_fp = tmp_path / "fatfraction_nb.nii.gz"
    f_fp = tmp_path / "fat.nii.gz"
    w_fp = tmp_path / "water.nii.gz"

    assert perror(ref_ff_nb_fp, ff_nb_fp) < pthresh
    assert perror(ref_f_fp, f_fp) < pthresh
    assert perror(ref_w_fp, w_fp) < pthresh


def test_musclemap_mtr_1(tmp_path, script_runner):

    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "mtr"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mt_on_fp = input_dir / "mt_on.nii.gz"
    mt_off_fp = input_dir / "mt_off.nii.gz"

    ref_mtr_fp = output_dir / "mtr.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME, "-o", str(tmp_path), "mtr", str(mt_on_fp), str(mt_off_fp)]
    )
    assert result.success

    mtr_fp = tmp_path / "mtr.nii.gz"

    assert perror(ref_mtr_fp, mtr_fp) < pthresh


def test_musclemap_mtr_quiet(tmp_path, script_runner):

    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "mtr"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mt_on_fp = input_dir / "mt_on.nii.gz"
    mt_off_fp = input_dir / "mt_off.nii.gz"

    ref_mtr_fp = output_dir / "mtr.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME, "-quiet", "-o", str(tmp_path), "mtr", str(mt_on_fp), str(mt_off_fp)]
    )
    assert result.success

    mtr_fp = tmp_path / "mtr.nii.gz"

    assert perror(ref_mtr_fp, mtr_fp) < pthresh


def test_musclemap_mtr_b1(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "mtr-b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mt_on_fp = input_dir / "mt_on.nii.gz"
    mt_off_fp = input_dir / "mt_off.nii.gz"

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"
    res_ref_fp = input_dir / "0006-Dixon_TE_345_th.nii.gz"

    ref_mtr_fp = output_dir / "mtr.nii.gz"
    ref_mtr_b1pcf_fp = output_dir / "mtr_b1pcf.nii.gz"
    ref_mtr_b1scf_fp = output_dir / "mtr_b1scf.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-o",
        str(tmp_path),
        "mtr-b1",
        str(mt_on_fp),
        str(mt_off_fp),
        str(fa60_fp),
        str(fa120_fp),
        str(res_ref_fp)]
    )
    assert result.success

    mtr_fp = tmp_path / "mtr.nii.gz"
    mtr_b1pcf_fp = tmp_path / "mtr_b1pcf.nii.gz"
    mtr_b1scf_fp = tmp_path / "mtr_b1scf.nii.gz"

    assert perror(ref_mtr_fp, mtr_fp) < pthresh
    assert perror(ref_mtr_b1pcf_fp, mtr_b1pcf_fp) < pthresh
    assert perror(ref_mtr_b1scf_fp, mtr_b1scf_fp) < pthresh


def test_musclemap_mtr_b1_quiet(tmp_path, script_runner):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "mtr-b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mt_on_fp = input_dir / "mt_on.nii.gz"
    mt_off_fp = input_dir / "mt_off.nii.gz"

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"
    res_ref_fp = input_dir / "0006-Dixon_TE_345_th.nii.gz"

    ref_mtr_fp = output_dir / "mtr.nii.gz"
    ref_mtr_b1pcf_fp = output_dir / "mtr_b1pcf.nii.gz"
    ref_mtr_b1scf_fp = output_dir / "mtr_b1scf.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-o",
        str(tmp_path),
        "-quiet",
        "mtr-b1",
        str(mt_on_fp),
        str(mt_off_fp),
        str(fa60_fp),
        str(fa120_fp),
        str(res_ref_fp)]
    )
    assert result.success

    mtr_fp = tmp_path / "mtr.nii.gz"
    mtr_b1pcf_fp = tmp_path / "mtr_b1pcf.nii.gz"
    mtr_b1scf_fp = tmp_path / "mtr_b1scf.nii.gz"

    assert perror(ref_mtr_fp, mtr_fp) < pthresh
    assert perror(ref_mtr_b1pcf_fp, mtr_b1pcf_fp) < pthresh
    assert perror(ref_mtr_b1scf_fp, mtr_b1scf_fp) < pthresh


def test_musclemap_mtr_b1_error(tmp_path, script_runner):

    data_dir = TEST_DATA_DIR / "mtr-b1"
    input_dir = data_dir / "input"

    mt_on_fp = input_dir / "mt_on.nii.gz"
    mt_off_fp = input_dir / "mt_off.nii.gz"

    fa60_fp = input_dir / "se_fa060.nii.gz"
    fa120_fp = input_dir / "se_fa120.nii.gz"
    res_ref_fp = input_dir / "0006-Dixon_TE_345_th.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-o",
        str(tmp_path),
        "-c",
        "0",
        "-1",
        "0",
        "-1",
        "0",
        "-1",
        "-quiet",
        "mtr-b1",
        str(mt_on_fp),
        str(mt_off_fp),
        str(fa60_fp),
        str(fa120_fp),
        str(res_ref_fp)]
    )
    assert not result.success
    assert result.stderr.startswith("* optional arguments -r, -c or -m cannot be used")


def test_musclemap_t2(tmp_path, script_runner):
    # When running on stoney with FSL 6.0.4 the co-registration step of the T2
    # map production is marginally different, but within a mask the images are
    # less than 2% different
    pthresh = 2.0

    data_dir = TEST_DATA_DIR / "t2"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    e1_fp = input_dir / "t2_te16ms.nii.gz"
    e2_fp = input_dir / "t2_te56ms.nii.gz"

    reg_ref_fp = input_dir / "0007-Dixon_TE_345_cf.nii.gz"
    ref_t2_fp = output_dir / "t2_m.nii.gz"
    ref_s0_fp = output_dir / "s0_m.nii.gz"
    mask_fp = output_dir / "mask.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-v",
        "-o",
        str(tmp_path),
        "t2",
        str(e1_fp),
        str(e2_fp),
        "16.0",
        "56.0",
        str(reg_ref_fp)]
    )
    assert result.success

    t2_fp = tmp_path / "t2.nii.gz"
    s0_fp = tmp_path / "s0.nii.gz"

    fp_dict = {"t2_fp": t2_fp, "s0_fp": s0_fp}

    fp_dict, to_delete = preproc.mask(fp_dict, mask_fp, tmp_path, [], FSL_DIR, False)

    assert perror(ref_t2_fp, fp_dict["t2_fp"]) < pthresh
    assert perror(ref_s0_fp, fp_dict["s0_fp"]) < pthresh


def test_musclemap_t2_quiet(tmp_path, script_runner):
    data_dir = TEST_DATA_DIR / "t2"
    input_dir = data_dir / "input"

    # test case when output directory exists
    out_dir = tmp_path / "t2"
    out_dir.mkdir()

    e1_fp = input_dir / "t2_te16ms.nii.gz"
    e2_fp = input_dir / "t2_te56ms.nii.gz"

    reg_ref_fp = input_dir / "0007-Dixon_TE_345_cf.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-o",
        str(out_dir),
        "-v",
        "-quiet",
        "t2",
        str(e1_fp),
        str(e2_fp),
        "16.0",
        "56.0",
        str(reg_ref_fp)]
    )
    assert result.success


def test_musclemap_t2_error(tmp_path, script_runner):
    data_dir = TEST_DATA_DIR / "t2"
    input_dir = data_dir / "input"

    e1_fp = input_dir / "t2_te16ms.nii.gz"
    e2_fp = input_dir / "t2_te56ms.nii.gz"

    reg_ref_fp = input_dir / "0007-Dixon_TE_345_cf.nii.gz"

    result = script_runner.run(
        [SCRIPT_NAME,
        "-c",
        "0",
        "-1",
        "0",
        "-1",
        "0",
        "-1",
        "-o",
        str(tmp_path),
        "t2",
        str(e1_fp),
        str(e2_fp),
        "16.0",
        "56.0",
        str(reg_ref_fp)]
    )
    assert not result.success
    assert result.stderr.startswith("* optional arguments -r, -c or -m cannot be used")

import pathlib

import nibabel as nib
import numpy as np
import pytest

import musclemap.preproc as preproc
import musclemap.vercheck as vercheck

THIS_DIR = pathlib.Path(__file__).resolve().parent
TEST_DATA_DIR = THIS_DIR / "test_data"
FSL_DIR = vercheck.get_fsldir()


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


def test_check_files_exist_1(tmp_path, capsys):

    a_fp = tmp_path / "a"
    b_fp = tmp_path / "b"

    a_fp.touch()
    b_fp.touch()

    fp_dict = {"a_fp": a_fp, "b_fp": b_fp}

    preproc.check_files_exist(fp_dict)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_check_files_exist_2(tmp_path, capsys):

    a_fp = tmp_path / "a"
    b_fp = tmp_path / "b"
    test_fp_dict = {"a_fp": a_fp, "b_fp": b_fp}

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        preproc.check_files_exist(test_fp_dict)

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR: %s/a does not exist, exiting\n" % str(tmp_path)


def test_check_shape_and_orientation_1(capsys):

    data_dir = TEST_DATA_DIR / "b1/input"
    a_fp = data_dir / "se_fa060.nii.gz"
    b_fp = data_dir / "se_fa120.nii.gz"
    fp_dict = {"a": a_fp, "b": b_fp}
    ref_fp = data_dir / "se_fa060.nii.gz"
    preproc.check_shape_and_orientation(fp_dict, ref_fp)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_check_shape_and_orientation_2(capsys):
    # This checks for mis-matched matrix sizes

    a_fp = TEST_DATA_DIR / "b1" / "input" / "se_fa060.nii.gz"
    b_fp = TEST_DATA_DIR / "b1" / "output" / "b1_c.nii.gz"
    fp_dict = {"a": a_fp, "b": b_fp}
    ref_fp = TEST_DATA_DIR / "b1" / "input" / "se_fa060.nii.gz"

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        preproc.check_shape_and_orientation(fp_dict, ref_fp)

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR: %s mismatched geometry\n" % str(b_fp)


def test_check_shape_and_orientation_3(capsys):
    # This checks for mis-matched affines (s- or q-forms)

    a_fp = TEST_DATA_DIR / "b1" / "input" / "se_fa060.nii.gz"
    b_fp = TEST_DATA_DIR / "mtr" / "input" / "mt_on.nii.gz"
    fp_dict = {"a": a_fp, "b": b_fp}
    ref_fp = TEST_DATA_DIR / "b1" / "input" / "se_fa060.nii.gz"

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        preproc.check_shape_and_orientation(fp_dict, ref_fp)

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR: %s mismatched geometry\n" % str(b_fp)


@pytest.mark.parametrize(
    "fp, expected_output",
    [
        (pathlib.Path("a.nii.gz"), pathlib.Path("a")),
        (pathlib.Path("b.nii"), pathlib.Path("b")),
        (pathlib.Path("c"), pathlib.Path("c")),
    ],
)
def test_remove_file_ext(fp, expected_output):
    assert preproc.remove_file_ext(fp) == expected_output


def test_unscale(tmp_path):

    ph_fp = (
        TEST_DATA_DIR
        / "ff"
        / "philips"
        / "thigh_dcm2niix"
        / "input"
        / "1801-Dixon_TE_345_th_ph.nii"
    )
    fp_dict = {"ph_fp": ph_fp}

    nii_test_in = nib.load(str(fp_dict["ph_fp"]))
    assert nii_test_in.dataobj.slope != 1
    assert nii_test_in.dataobj.inter != 0

    fp_dict, to_delete = preproc.unscale(fp_dict, tmp_path, [])
    assert to_delete == [fp_dict["ph_fp"]]

    nii_test_out = nib.load(str(fp_dict["ph_fp"]))
    assert nii_test_out.dataobj.slope == 1
    assert nii_test_out.dataobj.inter == 0


def test_register(tmp_path):
    pthresh = 1.0

    fa60_fp = TEST_DATA_DIR / "b1" / "input" / "se_fa060.nii.gz"
    fa120_fp = TEST_DATA_DIR / "b1" / "input" / "se_fa120.nii.gz"
    ref_reg_fp = TEST_DATA_DIR / "b1" / "output" / "se_fa120_r.nii.gz"

    fp_dict = {"fa60_fp": fa60_fp, "fa120_fp": fa120_fp}

    fp_dict, to_delete = preproc.register(
        fp_dict, fp_dict["fa60_fp"], tmp_path, [], FSL_DIR, False
    )

    assert to_delete == [fp_dict["fa120_fp"]]
    assert perror(ref_reg_fp, fp_dict["fa120_fp"]) < pthresh


def test_register_quiet(tmp_path):
    pthresh = 1.0

    fa60_fp = TEST_DATA_DIR / "b1" / "input" / "se_fa060.nii.gz"
    fa120_fp = TEST_DATA_DIR / "b1" / "input" / "se_fa120.nii.gz"
    ref_reg_fp = TEST_DATA_DIR / "b1" / "output" / "se_fa120_r.nii.gz"

    fp_dict = {"fa60_fp": fa60_fp, "fa120_fp": fa120_fp}

    fp_dict, to_delete = preproc.register(
        fp_dict, fp_dict["fa60_fp"], tmp_path, [], FSL_DIR
    )

    assert to_delete == [fp_dict["fa120_fp"]]
    assert perror(ref_reg_fp, fp_dict["fa120_fp"]) < pthresh


def test_register_dixon(tmp_path):
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

    fp_dict = {
        "mminus1_fp": mminus1_fp,
        "phiminus1_fp": phiminus1_fp,
        "m0_fp": m0_fp,
        "phi0_fp": phi0_fp,
        "m1_fp": m1_fp,
        "phi1_fp": phi1_fp,
    }

    m0_ref_fp = output_dir / "0008-Dixon_TE_460_hand_r.nii.gz"
    phi0_ref_fp = output_dir / "0009-Dixon_TE_460_hand_r.nii.gz"
    m1_ref_fp = output_dir / "0010-Dixon_TE_575_hand_r.nii.gz"
    phi1_ref_fp = output_dir / "0011-Dixon_TE_575_hand_r.nii.gz"

    fp_ref_dict = {
        "mminus1_fp": mminus1_fp,
        "phiminus1_fp": phiminus1_fp,
        "m0_fp": m0_ref_fp,
        "phi0_fp": phi0_ref_fp,
        "m1_fp": m1_ref_fp,
        "phi1_fp": phi1_ref_fp,
    }

    fp_dict, to_delete = preproc.register_dixon(fp_dict, tmp_path, [], FSL_DIR, False)

    for key in fp_dict:
        assert perror(fp_ref_dict[key], fp_dict[key]) < pthresh


def test_register_t2(tmp_path):

    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "t2"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    e1_fp = input_dir / "t2_te16ms.nii.gz"
    e2_fp = input_dir / "t2_te56ms.nii.gz"

    fp_dict = {"e1_fp": e1_fp, "e2_fp": e2_fp}

    reg_ref_fp = input_dir / "0007-Dixon_TE_345_cf.nii.gz"

    ref_e1_r_fp = output_dir / "t2_te16ms_r.nii.gz"
    ref_e2_r_fp = output_dir / "t2_te56ms_r.nii.gz"
    fp_ref_dict = {"e1_fp": ref_e1_r_fp, "e2_fp": ref_e2_r_fp}

    fp_dict, to_delete = preproc.register_t2(
        fp_dict, reg_ref_fp, tmp_path, [], FSL_DIR, False
    )

    for key in fp_dict:
        assert perror(fp_ref_dict[key], fp_dict[key]) < pthresh


def test_mask(tmp_path):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/ge/thigh_masked"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mask_fp = input_dir / "mask.nii.gz"

    mminus1_fp = input_dir / "40006-ORIG_DIXON_TE_345_TH-0000.nii.gz"
    phiminus1_fp = input_dir / "40006-ORIG_DIXON_TE_345_TH-0001.nii.gz"

    fp_dict = {"mminus1_fp": mminus1_fp, "phiminus1_fp": phiminus1_fp}

    mminus1_ref_fp = output_dir / "40006-ORIG_DIXON_TE_345_TH-0000_m.nii.gz"
    phiminus1_ref_fp = output_dir / "40006-ORIG_DIXON_TE_345_TH-0001_m.nii.gz"

    fp_ref_dict = {"mminus1_fp": mminus1_ref_fp, "phiminus1_fp": phiminus1_ref_fp}

    fp_dict, to_delete = preproc.mask(fp_dict, mask_fp, tmp_path, [], FSL_DIR, False)

    for key in fp_dict:
        assert perror(fp_ref_dict[key], fp_dict[key]) < pthresh


def test_mask_quiet(tmp_path):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "ff/ge/thigh_masked"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mask_fp = input_dir / "mask.nii.gz"

    mminus1_fp = input_dir / "40006-ORIG_DIXON_TE_345_TH-0000.nii.gz"
    phiminus1_fp = input_dir / "40006-ORIG_DIXON_TE_345_TH-0001.nii.gz"

    fp_dict = {"mminus1_fp": mminus1_fp, "phiminus1_fp": phiminus1_fp}

    mminus1_ref_fp = output_dir / "40006-ORIG_DIXON_TE_345_TH-0000_m.nii.gz"
    phiminus1_ref_fp = output_dir / "40006-ORIG_DIXON_TE_345_TH-0001_m.nii.gz"

    fp_ref_dict = {"mminus1_fp": mminus1_ref_fp, "phiminus1_fp": phiminus1_ref_fp}

    fp_dict, to_delete = preproc.mask(fp_dict, mask_fp, tmp_path, [], FSL_DIR)

    for key in fp_dict:
        assert perror(fp_ref_dict[key], fp_dict[key]) < pthresh


def test_crop(tmp_path):

    pthresh = 1.0
    crop_dims = [0, 56, 0, -1, 0, -1]

    data_dir = TEST_DATA_DIR / "b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    a_fp = input_dir / "se_fa060.nii.gz"
    b_fp = input_dir / "se_fa120.nii.gz"

    fp_dict = {"a": a_fp, "b": b_fp}

    a_ref_fp = output_dir / "se_fa060_c.nii.gz"
    b_ref_fp = output_dir / "se_fa120_c.nii.gz"

    fp_ref_dict = {"a": a_ref_fp, "b": b_ref_fp}

    fp_dict, to_delete = preproc.crop(fp_dict, crop_dims, tmp_path, [], FSL_DIR, False)

    for key in fp_dict:
        assert perror(fp_ref_dict[key], fp_dict[key]) < pthresh


def test_resample(tmp_path):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "mtr-b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    mt_on_fp = input_dir / "mt_on.nii.gz"
    mt_off_fp = input_dir / "mt_off.nii.gz"

    fp_dict = {"mt_on_fp": mt_on_fp, "mt_off_fp": mt_off_fp}

    res_ref_fp = input_dir / "0006-Dixon_TE_345_th.nii.gz"

    ref_mt_on_fp = output_dir / "mt_on_resamp.nii.gz"
    ref_mt_off_fp = output_dir / "mt_off_resamp.nii.gz"

    fp_ref_dict = {"mt_on_fp": ref_mt_on_fp, "mt_off_fp": ref_mt_off_fp}

    fp_dict, to_delete = preproc.resample(
        fp_dict, res_ref_fp, tmp_path, [], FSL_DIR, False
    )

    for key in fp_dict:
        assert perror(fp_ref_dict[key], fp_dict[key]) < pthresh


def test_create_mask(tmp_path):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / "mtr-b1"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    res_ref_fp = input_dir / "0006-Dixon_TE_345_th.nii.gz"

    ref_bgmask_fp = output_dir / "bgmask.nii.gz"

    bg_mask_fp, to_delete = preproc.create_mask(
        res_ref_fp, tmp_path, [], FSL_DIR, False
    )

    assert perror(ref_bgmask_fp, bg_mask_fp) < pthresh

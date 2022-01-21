import nibabel as nib
import numpy as np
import pathlib
import pytest

import musclemap.ff as ff
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


@pytest.mark.parametrize('phase, scanner, expected_output',
                         [(np.array([1000.0,  2000.0, -1000.0, 200.0]),
                           'ge',
                           np.array([1.0, 2.0, -1.0, 0.2])),
                          (np.array([2048.0,  4096.0, -2048.0, 1024.0]),
                           'siemens',
                           np.array([np.pi, 2.0 * np.pi, -np.pi, 0.5 * np.pi]))])
def test_scale_phase(phase, scanner, expected_output):
    assert np.allclose(ff.scale_phase(phase, scanner), expected_output)


@pytest.mark.parametrize('mag, phase, expected_output',
                         [(1.0, np.pi / 2.0, 1j),
                          (1.0, 0.0, 1.0),
                          (1.0, np.pi, -1.0),
                          (0.0, np.pi, 0.0),
                          (np.array([0.0, 5.0]),
                           np.array([np.pi, 3 * np.pi / 2.0]),
                           np.array([0.0, -5j]))])
def test_make_complex(mag, phase, expected_output):
    assert np.allclose(ff.make_complex(mag, phase), expected_output)


@pytest.mark.parametrize('c, phase, expected_output',
                         [(1.0 + 1.0j, np.pi / 4.0, np.sqrt(2) + 0.0j),
                          (1.0 + 0.0j, np.pi, -1.0 + 0.0j),
                          (np.array([-1.0 - 1.0j, 5.0]),
                           np.array([np.pi / 4.0, np.pi / 2.0]),
                           np.array([-np.sqrt(2), -5j]))])
def test_subtract_phase(c, phase, expected_output):
    assert np.allclose(ff.subtract_phase(c, phase), expected_output)


@pytest.mark.parametrize('z1, z2, expected_output',
                         [(1.0 + 1.0j, 1.0 + 1.0j, 0.0),
                          (1.0 + 0.0j, -1.0 - 1.0j, 3.0 * np.pi / 4.0),
                          (0.0 - 1.0j, -5.0 - 0.0j, np.pi / 2.0),
                          (np.array([1.0 + 1.0j, 1.0 + 0.0j, 0.0 - 1.0j]),
                           np.array([1.0 + 1.0j, -1.0 - 1.0j, -5.0 - 0.0j]),
                           np.array([0.0, 3.0 * np.pi / 4.0, np.pi / 2.0]))])
def test_calc_phim(z1, z2, expected_output):
    assert np.allclose(ff.calc_phim(z1, z2), expected_output)


def test_unwrap(tmp_path):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / 'ff/siemens/thigh'
    input_dir = data_dir / 'input'
    output_dir = data_dir / 'output'

    ref_phim_uw_fp = output_dir / 'phim_uw.nii.gz'

    phim_fp = output_dir / 'phim.nii.gz'
    phim_shape = nib.load(str(phim_fp)).header.get_data_shape()

    m0_fp = input_dir / '0008-Dixon_TE_460_th.nii.gz'

    phim_uw_fp, to_delete = ff.unwrap(phim_fp, phim_shape, m0_fp, tmp_path, [],
                                   True, vercheck.get_fsldir(), False)

    assert perror(ref_phim_uw_fp, phim_uw_fp) < pthresh


@pytest.mark.parametrize('test_phim_uw, test_s, expected_output',
                         [(np.array([0.0, np.pi / 2.0, np.pi, np.pi, np.pi]),
                           np.array([1.0 + 0.0j, 1.0 + 1.0j, 0.0 + 1.0j,
                                     -1.0 + 0.0j, 0.0 - 1.0j]),
                           np.array([1.0, 0.0, -1.0, 0.0, 1.0]))])
def test_calc_p(test_phim_uw, test_s, expected_output):
    assert np.allclose(ff.calc_p(test_phim_uw, test_s), expected_output)


@pytest.mark.parametrize('test_sminus1prime, test_s0prime, test_s1prime,'
                         'test_p, expected_output',
                         [(5.0, 16.0 + 63.0j, 5.0, -1.0, 30.0),
                          (3.0, 44.0 + 117.0j, 3.0, -0.5, 61.75),
                          (10.0, 44.0 + 117.0j, 10.0, 0.0, 62.5),
                          (10.0, 44.0 + 117.0j, 10.0, 1.0, 67.5),
                          (np.array([5.0, 3.0, 10.0, 10.0]),
                           np.array([16.0 + 63.0j, 44.0 + 117.0j,
                                     44.0 + 117.0j, 44.0 + 117.0j]),
                           np.array([5.0, 3.0, 10.0, 10.0]),
                           np.array([-1.0, -0.5, 0.0, 1.0]),
                           np.array([30.0, 61.75, 62.5, 67.5]))])
def test_calc_water(test_sminus1prime, test_s0prime, test_s1prime, test_p,
                    expected_output):
    assert np.allclose(ff.calc_water(test_sminus1prime, test_s0prime,
                                     test_s1prime, test_p), expected_output)


@pytest.mark.parametrize('test_sminus1prime, test_s0prime, test_s1prime,'
                         'test_p, expected_output',
                         [(5.0, 16.0 + 63.0j, 5.0, -1.0, 35.0),
                          (3.0, 44.0 + 117.0j, 3.0, -0.5, 63.25),
                          (10.0, 44.0 + 117.0j, 10.0, 0.0, 62.5),
                          (10.0, 44.0 + 117.0j, 10.0, 1.0, 57.5),
                          (np.array([5.0, 3.0, 10.0, 10.0]),
                           np.array([16.0 + 63.0j, 44.0 + 117.0j,
                                     44.0 + 117.0j, 44.0 + 117.0j]),
                           np.array([5.0, 3.0, 10.0, 10.0]),
                           np.array([-1.0, -0.5, 0.0, 1.0]),
                           np.array([35.0, 63.25, 62.5, 57.5]))])
def test_calc_fat(test_sminus1prime, test_s0prime, test_s1prime, test_p,
                  expected_output):
    assert np.allclose(ff.calc_fat(test_sminus1prime, test_s0prime,
                                   test_s1prime, test_p), expected_output)


@pytest.mark.parametrize('test_water, test_fat, expected_output',
                         [(np.array([1.0, 0.0, 1.0]),
                           np.array([1.0, 1.0, 0.0]),
                           np.array([50.0, 100.0, 0.0]))])
def test_calc_ff(test_water, test_fat, expected_output):
    assert np.allclose(ff.calc_ff(test_water, test_fat), expected_output)


@pytest.mark.parametrize('test_sminus1prime, test_s0prime, test_s1prime,'
                         'test_p, expected_output',
                         [(np.array([5.0, 3.0, 10.0, 10.0, 0.0]),
                           np.array([16.0 + 63.0j, 44.0 + 117.0j,
                                     44.0 + 117.0j, 44.0 + 117.0j,
                                     44.0 + 117.0j]),
                           np.array([5.0, 3.0, 10.0, 10.0, 0.0]),
                           np.array([-1.0, -0.5, 0.0, 1.0, 0.0]),
                           np.array([51.08294682, 50.21434058, 50.0,
                                     48.45584417, 50.0]))])
def test_calc_ff_nb(test_sminus1prime, test_s0prime, test_s1prime, test_p,
                    expected_output):
    assert np.allclose(ff.calc_ff_nb(test_sminus1prime, test_s0prime,
                                  test_s1prime, test_p), expected_output)


def test_process_ff(tmp_path):
    pthresh = 1.0

    data_dir = TEST_DATA_DIR / 'ff/siemens/thigh'
    input_dir = data_dir / 'input'
    output_dir = data_dir / 'output'

    mminus1_fp = input_dir / '0006-Dixon_TE_345_th.nii.gz'
    phiminus1_fp = input_dir / '0007-Dixon_TE_345_th.nii.gz'
    m0_fp = input_dir / '0008-Dixon_TE_460_th.nii.gz'
    phi0_fp = input_dir / '0009-Dixon_TE_460_th.nii.gz'
    m1_fp = input_dir / '0010-Dixon_TE_575_th.nii.gz'
    phi1_fp = input_dir / '0011-Dixon_TE_575_th.nii.gz'

    fp_dict = {'mminus1_fp': mminus1_fp, 'phiminus1_fp': phiminus1_fp,
               'm0_fp': m0_fp, 'phi0_fp': phi0_fp,
               'm1_fp': m1_fp, 'phi1_fp': phi1_fp}

    ref_ff_fp = output_dir / 'fatfraction.nii.gz'

    ff_fp, to_delete = ff.process_ff(fp_dict, tmp_path, [],
                                     vercheck.get_fsldir(), False,
                                     'siemens', True, False)

    assert perror(ref_ff_fp, ff_fp) < pthresh
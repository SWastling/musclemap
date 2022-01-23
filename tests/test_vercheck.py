import os
from unittest import mock

import pytest

import musclemap.config as config
import musclemap.vercheck as vercheck


def test_get_fsldir(capsys, tmp_path):

    # Check appropriate error is thrown if FSLDIR not set
    with mock.patch.dict(os.environ, clear=True):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            vercheck.get_fsldir()

        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert (
            captured.err == "ERROR: Environment variable FSLDIR is not "
            "set; please run appropriate FSL "
            "configuration script\n"
        )

    # Mock the FSLDIR environment variable
    with mock.patch.dict(os.environ, {"FSLDIR": str(tmp_path)}):
        assert vercheck.get_fsldir() == tmp_path


def test_get_fsl_ver(tmp_path):

    assert vercheck.get_fsl_ver(vercheck.get_fsldir()) in config.FSL_VERSIONS

    # Make a mock fslversion file and check can read it
    with mock.patch.dict(os.environ, {"FSLDIR": str(tmp_path)}):
        etc_dir = tmp_path / "etc"
        etc_dir.mkdir()

        # Check what happens if it can't find fslversion file
        fslversion_fp = etc_dir / "fslversion_wrong"
        fslversion_fp.write_text("1.0.0:abcde")
        fsldir = vercheck.get_fsldir()
        assert vercheck.get_fsl_ver(fsldir) == "unknown"


def test_check_lib_ver(capsys):

    # Library check fail and exit
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        vercheck.check_lib_ver("lib_a", "2.0.0", ["1.0.0", "1.0.1"], False)

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1
    captured = capsys.readouterr()
    assert (
        captured.out == "** FAIL using non-validated lib_a version\n*** "
        "expected 1.0.0 or 1.0.1, got 2.0.0\n"
    )
    assert captured.err == "** exiting\n"

    # Library check fail and don't exit
    vercheck.check_lib_ver("lib_a", "2.0.0", ["1.0.0", "1.0.1"], True)
    captured = capsys.readouterr()
    assert (
        captured.out == "** FAIL using non-validated lib_a version\n*** "
        "expected 1.0.0 or 1.0.1, got 2.0.0\n"
    )

    # Library check pass and don't exit
    vercheck.check_lib_ver("lib_a", "2.0.0", ["1.0.0", "2.0.0"], True)
    captured = capsys.readouterr()
    assert captured.out == "** PASS version check on lib_a (2.0.0)\n"

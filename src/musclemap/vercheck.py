import os
import pathlib
import sys


def get_fsldir():
    """
    Get the environment variable FSLDIR

    :return: fsldir
    :rtype fsldir: pathlib.Path
    """

    fsldir = os.getenv("FSLDIR")

    if fsldir is None:
        sys.stderr.write(
            "ERROR: Environment variable FSLDIR is not set; "
            "please run appropriate FSL configuration script\n"
        )
        sys.exit(1)
    else:
        return pathlib.Path(fsldir)


def get_fsl_ver(fsldir):
    """
    Determine FSL version installed on system based on environment variable
    FSLDIR

    :param fsldir: Directory set by FSLDIR environment variable
    :type fsldir: pathlib.Path
    :return: fslver
    :rtype fslver: str
    """

    fslversion_fp = fsldir / "etc" / "fslversion"

    try:
        fslver = fslversion_fp.read_text()
    except EnvironmentError:
        fslver = "unknown"

    return fslver


def check_lib_ver(lib_name, lib_ver, expected_lib_ver_list, any_ver):
    """
    Compare the actual library version to the expected version

    :param lib_name: name of library
    :type lib_name: str
    :param lib_ver: version of library
    :type lib_ver: str
    :param expected_lib_ver_list: expected version of library
    :type expected_lib_ver_list: list
    :param any_ver: don't abort if any_ver is True
    :rtype any_ver: bool
    """
    if lib_ver in expected_lib_ver_list:
        print("** PASS version check on %s (%s)" % (lib_name, lib_ver))
    else:
        print("** FAIL using non-validated %s version" % lib_name)
        print("*** expected %s, got %s" % (" or ".join(expected_lib_ver_list), lib_ver))

        if not any_ver:
            sys.stderr.write("** exiting\n")
            sys.exit(1)

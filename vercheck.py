import os
import pathlib
import re
import subprocess as sp
import sys


def git_version(srcdir):
    """
    Determine if we are running a release and if so what version

    :param srcdir: Directory containing git repo
    :type srcdir: pathlib.Path
    :return: ver, release
    """

    git_dir = srcdir / '.git'
    if git_dir.is_dir():
        sp_out = sp.run(['git', 'tag', '-l', 'release-*'], check=True,
                        capture_output=True, text=True, cwd=srcdir)

        tag_list = sp_out.stdout.strip()

        if tag_list:
            sp_out = sp.run(['git', 'describe', '--tags', '--long',
                             '--dirty=+'], check=True, capture_output=True,
                            text=True, cwd=srcdir)
            ver_long = sp_out.stdout.strip()
            ver_components = ver_long.split('-')

            # If a tag has been defined the output of git describe is
            # tag-commits_ontop-git_sha e.g.

            # Case 1: release-1.0.0-0-4609e1a
            # clean with no commits on top the tag
            # version = release-1.0.0 and release is True

            # Case 2: release-1.0.0-0-4609e1a+
            # dirty so release is False

            # Case 3: release-1.0.0-1-4609e1a
            # 1 commit on top the tag so release is False

            # Case 4 release-1.0.0-2-4609e1a+
            # 2 commits on top the tag and it's dirty so release is False

            git_sha = ver_components.pop()
            commits_ontop = int(ver_components.pop())
            ver = "-".join(ver_components)

            if re.match(r'release-', ver) and commits_ontop == 0 \
                    and git_sha[-1] != '+':
                release = True
            else:
                # Show user long version of tag so they can see why the version
                # check failed i.e. commits on top or dirty repo
                ver = ver + '-' + str(commits_ontop) + '-' + git_sha
                release = False
        else:
            # No tags with prefix release-
            ver = 'unknown'
            release = False
    else:
        # No .git directory
        ver = 'unknown'
        release = False

    return ver, release


def get_fsldir():
    """
    Get the environment variable FSLDIR

    :return: fsldir
    :rtype fsldir: pathlib.Path
    """

    fsldir = os.getenv('FSLDIR')

    if fsldir is None:
        sys.stderr.write('ERROR: Environment variable FSLDIR is not set; '
                         'please run appropriate FSL configuration script\n')
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

    fslversion_fp = fsldir / 'etc' / 'fslversion'

    try:
        fslver = fslversion_fp.read_text()
    except EnvironmentError:
        fslver = 'unknown'

    return fslver


def check_script_ver(script_fp, any_ver):
    """
    Determine if the script we are running is a release

    :param script_fp: name of library
    :type script_fp: pathlib.Path
    :param any_ver: don't abort if any_ver is True
    :rtype any_ver: bool
    """
    version, release = git_version(script_fp.resolve().parent)

    if release:
        print("** PASS release check on %s (%s)" % (script_fp.name, version))
    else:
        print("** FAIL %s not tagged as a clean release (version %s)"
              % (script_fp.name, version))

        if not any_ver:
            sys.stderr.write("** exiting\n")
            sys.exit(1)


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
        print("*** expected %s, got %s" % (' or '.join(expected_lib_ver_list),
                                           lib_ver))

        if not any_ver:
            sys.stderr.write("** exiting\n")
            sys.exit(1)

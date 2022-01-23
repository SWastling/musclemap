import subprocess as sp

import nibabel as nib
import numpy as np


def scale_phase(phase, scanner):
    """
    Scale phase to radians

    :param phase: Phase in scanner units
    :type phase: float or np.ndarray
    :param scanner: scanner type {ge, siemens}
    :type scanner: str
    :return: phase in radians
    """
    if scanner == "siemens":
        sf = np.pi / 2048.0
    elif scanner == "ge":
        sf = 1.0 / 1000.0

    return phase * sf


def make_complex(mag, phase):
    """
    Make a complex number from magnitude and phase

    :param mag: magnitude of complex number
    :param phase: phase of complex number in radians
    :return: complex number
    """
    return mag * np.exp(1j * phase)


def subtract_phase(c, phase):
    """
    Subtract phase from a complex number
    :param c: complex number
    :param phase: phase in radians
    :return: complex number with phase subtracted
    """
    return c * np.exp(-1j * phase)


def calc_phim(z1, z2):
    """
    Multiply a complex number by the complex conjugate of a second complex
    number and determine the phase angle of the result

    :param z1: complex number
    :param z2: complex number
    :return: phase of complex number z1 multiplied by complex conjugate of z2
    """
    return np.angle(z1 * z2.conjugate())


def unwrap(phim_fp, phim_shape, m0_fp, out_dir, to_delete, split, fsldir, quiet=True):

    """
    Unwrap phase image using FSL prelude

    :param phim_fp: phase image filepath
    :type phim_fp: pathlib.Path
    :param phim_shape: shape array
    :param m0_fp: magnitude image filepath
    :type m0_fp: pathlib.Path
    :param out_dir: directory to store phim_uw NIfTI file in
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :param split: split image in left-right direction
    :param fsldir: full path to FSL directory
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: phim_uw: unwrapped phase image
    """

    phim_uw_fp = out_dir / "phim_uw.nii.gz"

    if split:
        phim_lr_uw_fp_list = []
        for side in ["left", "right"]:
            if not quiet:
                print("**** %s side" % side)
            halfx = int(phim_shape[0] / 2)
            if side == "left":
                startx = "0"
                sizex = str(halfx)
            elif side == "right":
                startx = str(halfx)
                sizex = "-1"

            # Split the phase image in half
            phim_lr_fp = out_dir / ("phim_%s.nii.gz" % side)

            fslmaths_cmd = [
                str(fsldir / "bin" / "fslmaths"),
                str(phim_fp),
                "-roi",
                startx,
                sizex,
                "0",
                "-1",
                "0",
                "-1",
                "0",
                "-1",
                str(phim_lr_fp),
            ]

            if not quiet:
                print("*****", " ".join(fslmaths_cmd))

            sp.run(fslmaths_cmd, check=True, text=True)

            # Split the magnitude image in half
            m0_lr_fp = out_dir / ("m0_%s.nii.gz" % side)

            fslmaths_cmd = [
                str(fsldir / "bin" / "fslmaths"),
                str(m0_fp),
                "-roi",
                startx,
                sizex,
                "0",
                "-1",
                "0",
                "-1",
                "0",
                "-1",
                str(m0_lr_fp),
            ]

            if not quiet:
                print("*****", " ".join(fslmaths_cmd))

            sp.run(fslmaths_cmd, check=True, text=True)

            # Unwrap with prelude
            phim_lr_uw_fp = out_dir / ("phim_%s_uw.nii.gz" % side)

            phim_lr_uw_fp_list.append(phim_lr_uw_fp)

            prelude_cmd = [
                str(fsldir / "bin" / "prelude"),
                "-p",
                str(phim_lr_fp),
                "-a",
                str(m0_lr_fp),
                "-o",
                str(phim_lr_uw_fp),
            ]

            if not quiet:
                prelude_cmd.append("-v")
                print("*****", " ".join(prelude_cmd))

            sp.run(prelude_cmd, check=True, text=True)

            to_delete.extend([phim_lr_fp, m0_lr_fp, phim_lr_uw_fp])

        if not quiet:
            print("**** combining unwrapped images")

        fslmaths_cmd = [
            str(fsldir / "bin" / "fslmaths"),
            str(phim_lr_uw_fp_list[0]),
            "-add",
            str(phim_lr_uw_fp_list[1]),
            str(phim_uw_fp),
            "-odt",
            "float",
        ]

        if not quiet:
            print("****", " ".join(fslmaths_cmd))

        sp.run(fslmaths_cmd, check=True, text=True)

    else:
        prelude_cmd = [
            str(fsldir / "bin" / "prelude"),
            "-p",
            str(phim_fp),
            "-a",
            str(m0_fp),
            "-o",
            str(phim_uw_fp),
        ]

        if not quiet:
            prelude_cmd.append("-v")
            print("****", " ".join(prelude_cmd))

        sp.run(prelude_cmd, check=True, text=True)

    to_delete.append(phim_uw_fp)

    return phim_uw_fp, to_delete


def calc_p(phim_uw, sminus1prime):
    """
    Calculate the switch function p

    :param phim_uw: unwrapped phase image
    :param sminus1prime: fat-water out-of-phase complex image corrected for phi0
    :return: switch function p
    """
    # Eq 12 Glover and Schneider 1991
    theta = np.angle(sminus1prime * np.exp(1j * phim_uw / 2.0))
    return np.cos(theta)


def calc_water(sminus1prime, s0prime, s1prime, p):
    """
    Calculate water image using Eq 14 Glover and Schneider 1991

    :param sminus1prime: fat-water out-of-phase complex image corrected for phi0
    :param s0prime: fat-water in-phase complex image corrected for phi0
    :param s1prime: fat-water out-of-phase complex image corrected for phi0
    :param p: switch function
    :return: water image
    """
    return 0.5 * (abs(s0prime) + p * abs(np.sqrt(s1prime * sminus1prime)))


def calc_fat(sminus1prime, s0prime, s1prime, p):
    """
    Calculate fat image using Eq 14 Glover and Schneider 1991

    :param sminus1prime: fat-water out-of-phase complex image corrected for phi0
    :param s0prime: fat-water in-phase complex image corrected for phi0
    :param s1prime: fat-water out-of-phase complex image corrected for phi0
    :param p: switch function
    :return: fat image
    """
    return 0.5 * (abs(s0prime) - p * abs(np.sqrt(s1prime * sminus1prime)))


def calc_ff(w, f):
    """
    Calculate fat-fraction

    :param w: water image
    :param f: fat image
    :return: fat-fraction as percentage
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ff = 100 * f / (w + f)
        ff = np.nan_to_num(ff, copy=False, posinf=0.0, neginf=0.0)

    return ff


def calc_ff_nb(sminus1prime, s0prime, s1prime, p):
    """
    Calculate noise bias correct fat-fraction image using
    Liu Magnitude discrimination MRM 2007

    :param sminus1prime: fat-water out-of-phase complex image corrected for phi0
    :param s0prime: fat-water in-phase complex image corrected for phi0
    :param s1prime: fat-water out-of-phase complex image corrected for phi0
    :param p: switch function
    :return: fat-fraction with noise-bias correction image
    """
    rho1dc = 0.5 * (s0prime + p * np.sqrt(s1prime * sminus1prime))
    rho2dc = 0.5 * (s0prime - p * np.sqrt(s1prime * sminus1prime))

    complex_denominator = rho1dc + rho2dc

    with np.errstate(divide="ignore", invalid="ignore"):
        # fat dominant condition
        ff_nb_a = 100 * (abs(rho2dc) / abs(complex_denominator))
        ff_nb_a = np.nan_to_num(ff_nb_a, copy=False, posinf=0.0, neginf=0.0)
        # water dominant condition
        ff_nb_b = 100 * (1 - (abs(rho1dc) / abs(complex_denominator)))
        ff_nb_b = np.nan_to_num(ff_nb_b, copy=False, posinf=0.0, neginf=0.0)

    ff_nb = ff_nb_b  # default to water dominant

    # water dominant
    ff_nb[np.where(rho1dc >= rho2dc)] = ff_nb_b[np.where(rho1dc >= rho2dc)]
    # fat dominant
    ff_nb[np.where(rho1dc < rho2dc)] = ff_nb_a[np.where(rho1dc < rho2dc)]

    return ff_nb


def process_ff(
    fp_dict, out_dir, to_delete, fsldir, noise_bias, scanner, split, quiet=True
):

    """
    Calculate fat-fraction map from 3 point Dixon data

    :param fp_dict: filenames of in- and out-of-phase NIfTI files
    :type fp_dict: dict
    :param out_dir: directory to store fat-fraction map NIfTI file in
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :type to_delete: list
    :param fsldir: full path to FSL directory
    :type fsldir: pathlib.Path
    :param noise_bias: calculate noise-bias corrected fat-fraction map
    :type noise_bias: bool
    :param scanner: scanner type {ge or siemens}
    :type scanner: str
    :param split: split image in left-right direction
    :type split: bool
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: ff_fp, to_delete: fat-fraction filepath(s) and intermediate
    files to delete
    :rtype: tuple
    """

    if not quiet:
        print("** loading NIfTI image data")
    mminus1 = nib.load(str(fp_dict["mminus1_fp"])).get_fdata()
    phiminus1 = nib.load(str(fp_dict["phiminus1_fp"])).get_fdata()
    m0 = nib.load(str(fp_dict["m0_fp"])).get_fdata()
    phi0 = nib.load(str(fp_dict["phi0_fp"])).get_fdata()
    m1 = nib.load(str(fp_dict["m1_fp"])).get_fdata()
    phi1 = nib.load(str(fp_dict["phi1_fp"])).get_fdata()

    affine_out = nib.load(str(fp_dict["mminus1_fp"])).header.get_best_affine()

    if not quiet:
        print("** scaling phase images to radians")
    phiminus1_rad = scale_phase(phiminus1, scanner)
    phi0_rad = scale_phase(phi0, scanner)
    phi1_rad = scale_phase(phi1, scanner)

    if not quiet:
        print("** combining magnitude and phase images to make s-1, s0 and s1")
    # Eq.2-4 from Glover and Schneider 1991
    sminus1 = make_complex(mminus1, phiminus1_rad)
    s0 = make_complex(m0, phi0_rad)
    s1 = make_complex(m1, phi1_rad)

    if not quiet:
        print("** calculating s_1prime, s0prime and s1prime by subtracting phi0")
    sminus1prime = subtract_phase(sminus1, phi0_rad)
    s0prime = subtract_phase(s0, phi0_rad)
    s1prime = subtract_phase(s1, phi0_rad)

    if not quiet:
        print("** calculating phim")
    phim = calc_phim(s1prime, sminus1prime)

    if not quiet:
        print("*** saving NIfTI file")

    phim_fp = out_dir / "phim.nii.gz"
    phim_nii_obj = nib.nifti1.Nifti1Image(phim, affine_out)
    phim_nii_obj.to_filename(str(phim_fp))

    if not quiet:
        print("*** unwrapping phim with prelude")

    phim_uw_fp, to_delete = unwrap(
        phim_fp,
        np.shape(phim),
        fp_dict["m0_fp"],
        out_dir,
        to_delete,
        split,
        fsldir,
        quiet,
    )

    if not quiet:
        print("**** loading NIfTI image data")
    phim_uw = nib.load(str(phim_uw_fp)).get_fdata()

    if not quiet:
        print("** calculating p")
    p = calc_p(phim_uw, sminus1prime)

    if not quiet:
        print("** calculating water and fat images")
    w = calc_water(sminus1prime, s0prime, s1prime, p)
    f = calc_fat(sminus1prime, s0prime, s1prime, p)

    if not quiet:
        print("** calculating fat-fraction map")
    ff = calc_ff(w, f)

    if not quiet:
        print("** saving NIfTI files")
    w_fp = out_dir / "water.nii.gz"
    w_obj = nib.nifti1.Nifti1Image(w, affine_out)
    w_obj.to_filename(str(w_fp))

    f_fp = out_dir / "fat.nii.gz"
    f_obj = nib.nifti1.Nifti1Image(f, affine_out)
    f_obj.to_filename(str(f_fp))

    ff_fp = out_dir / "fatfraction.nii.gz"
    ff_obj = nib.nifti1.Nifti1Image(ff, affine_out)
    ff_obj.to_filename(str(ff_fp))

    if noise_bias:
        if not quiet:
            print("** calculating noise-bias corrected fat-fraction map")
        ff_nb = calc_ff_nb(sminus1prime, s0prime, s1prime, p)

        if not quiet:
            print("** saving NIfTI file")
        ff_nb_fp = out_dir / "fatfraction_nb.nii.gz"
        ff_nb_obj = nib.nifti1.Nifti1Image(ff_nb, affine_out)
        ff_nb_obj.to_filename(str(ff_nb_fp))

        return [ff_fp, ff_nb_fp], to_delete
    else:
        return ff_fp, to_delete

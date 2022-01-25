import subprocess as sp
import sys

import nibabel as nib


def check_files_exist(fp_dict):
    """
    Check if a set of file paths exist

    :param fp_dict: dictionary of NIfTI filenames (as pathlib.Path objects)
    :type fp_dict: dict
    """

    for fp in fp_dict.values():
        if not fp.is_file():
            sys.stderr.write("ERROR: %s does not exist, exiting\n" % str(fp))
            sys.exit(1)


def check_shape_and_orientation(fp_dict, ref_fp):
    """
    Compare the affine and matrix size in the header of a set of NIfTI files
    to a reference

    :param fp_dict: dictionary of NIfTI files (as pathlib.Path objects)
    :type fp_dict: dict
    :param ref_fp: reference NIfTI file
    :type ref_fp: pathlib.Path
    """

    # Define the reference image against which the other will be compared
    ref_obj = nib.load(str(ref_fp))
    ref_affine = ref_obj.header.get_best_affine()
    ref_shape = ref_obj.header.get_data_shape()

    # Loop over the input images comparing the affine in the NIfTI header
    for key, fp in fp_dict.items():
        if fp == ref_fp:
            continue
        else:
            check_obj = nib.load(str(fp))
            check_affine = check_obj.header.get_best_affine()
            check_shape = check_obj.header.get_data_shape()
            if not ((ref_affine == check_affine).all() and (ref_shape == check_shape)):
                sys.stderr.write("ERROR: %s mismatched geometry\n" % str(fp))
                sys.exit(1)


def remove_file_ext(fp):

    """
    Remove the extension(s) from the end of a filepath e.g. nii or nii.gz

    :param fp: filepath with nii or nii.gz extension
    :type fp: pathlib.Path
    :return: filepath without nii or nii.gz extension
    :rtype: pathlib.Path
    """

    while fp.suffixes:
        fp = fp.with_suffix("")

    return fp


def register(fp_dict, ref_fp, out_dir, to_delete, fsldir, quiet=True):
    """
    Register a set of NIfTI files to a reference with FSL flirt
    (rigid-body 6 degrees of freedom, correlation ratio cost-function)

    :param fp_dict: dictionary of NIfTI files
    :type fp_dict: dict
    :param ref_fp: reference NIfTI file
    :type ref_fp: pathlib.Path
    :param out_dir: output directory
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :type to_delete: list
    :param fsldir: FSL directory
    :type fsldir: pathlib.Path
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: fp_dict, to_delete
    :rtype: tuple
    """

    for key, fp in fp_dict.items():
        if fp == ref_fp:
            continue
        else:
            r_fn = remove_file_ext(fp).name + "_r.nii.gz"
            r_fp = out_dir / r_fn
            # Keep a list of intermediate files to potentially delete
            to_delete.append(r_fp)

            flirt_cmd = [
                str(fsldir / "bin" / "flirt"),
                "-cost",
                "corratio",
                "-dof",
                "6",
                "-in",
                str(fp),
                "-ref",
                str(ref_fp),
                "-out",
                str(r_fp),
            ]

            if not quiet:
                print("***", " ".join(flirt_cmd))

            # Capture the output from flirt as user doesn't need to see it
            sp.run(flirt_cmd, check=True, text=True, capture_output=True)

            fp_dict[key] = r_fp

    return fp_dict, to_delete


def register_dixon(fp_dict, out_dir, to_delete, fsldir, quiet=True):
    """
    Register a set of magnitude NIfTI files to a reference with FSL flirt
    (rigid-body 6 degrees of freedom, correlation ratio cost-function)

    Apply the relevant transformation matrix to phase images

    :param fp_dict: dictionary of NIfTI files
    :type fp_dict: dict
    :param out_dir: output directory
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :type to_delete: list
    :param fsldir: FSL directory
    :type fsldir: pathlib.Path
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: fp_dict, to_delete
    :rtype: tuple
    """

    ref_fp = fp_dict["mminus1_fp"]

    # Register m0 image to reference
    base_fn = remove_file_ext(fp_dict["m0_fp"]).name
    m0_r_fn = base_fn + "_r.nii.gz"
    m0_r_fp = out_dir / m0_r_fn
    m0_omat_fn = base_fn + "_xform.mat"
    m0_omat_fp = out_dir / m0_omat_fn

    flirt_cmd = [
        str(fsldir / "bin" / "flirt"),
        "-cost",
        "corratio",
        "-dof",
        "6",
        "-in",
        str(fp_dict["m0_fp"]),
        "-ref",
        str(ref_fp),
        "-out",
        str(m0_r_fp),
        "-omat",
        str(m0_omat_fp),
    ]

    if not quiet:
        print("***", " ".join(flirt_cmd))

    # Capture the output from flirt as user doesn't need to see it
    sp.run(flirt_cmd, check=True, text=True, capture_output=True)

    # Register phi0 image to reference (by applying xform from m0)
    base_fn = remove_file_ext(fp_dict["phi0_fp"]).name
    phi0_r_fn = base_fn + "_r.nii.gz"
    phi0_r_fp = out_dir / phi0_r_fn

    flirt_cmd = [
        str(fsldir / "bin" / "flirt"),
        "-in",
        str(fp_dict["phi0_fp"]),
        "-ref",
        str(ref_fp),
        "-out",
        str(phi0_r_fp),
        "-applyxfm",
        "-init",
        str(m0_omat_fp),
    ]

    if not quiet:
        print("***", " ".join(flirt_cmd))

    # Capture the output from flirt as user doesn't need to see it
    sp.run(flirt_cmd, check=True, text=True, capture_output=True)

    # Register m1 image to reference
    base_fn = remove_file_ext(fp_dict["m1_fp"]).name
    m1_r_fn = base_fn + "_r.nii.gz"
    m1_r_fp = out_dir / m1_r_fn
    m1_omat_fn = base_fn + "_xform.mat"
    m1_omat_fp = out_dir / m1_omat_fn

    flirt_cmd = [
        str(fsldir / "bin" / "flirt"),
        "-cost",
        "corratio",
        "-dof",
        "6",
        "-in",
        str(fp_dict["m1_fp"]),
        "-ref",
        str(ref_fp),
        "-out",
        str(m1_r_fp),
        "-omat",
        str(m1_omat_fp),
    ]

    if not quiet:
        print("***", " ".join(flirt_cmd))

    # Capture the output from flirt as user doesn't need to see it
    sp.run(flirt_cmd, check=True, text=True, capture_output=True)

    # Register phi1 image to reference (by applying xform from m1)
    base_fn = remove_file_ext(fp_dict["phi1_fp"]).name
    phi1_r_fn = base_fn + "_r.nii.gz"
    phi1_r_fp = out_dir / phi1_r_fn

    flirt_cmd = [
        str(fsldir / "bin" / "flirt"),
        "-in",
        str(fp_dict["phi1_fp"]),
        "-ref",
        str(ref_fp),
        "-out",
        str(phi1_r_fp),
        "-applyxfm",
        "-init",
        str(m1_omat_fp),
    ]

    if not quiet:
        print("***", " ".join(flirt_cmd))

    # Capture the output from flirt as user doesn't need to see it
    sp.run(flirt_cmd, check=True, text=True, capture_output=True)

    fp_dict["m0_fp"] = m0_r_fp
    fp_dict["phi0_fp"] = phi0_r_fp
    fp_dict["m1_fp"] = m1_r_fp
    fp_dict["phi1_fp"] = phi1_r_fp

    to_delete.extend([m0_r_fp, phi0_r_fp, m1_r_fp, phi1_r_fp, m0_omat_fp, m1_omat_fp])

    return fp_dict, to_delete


def register_t2(fp_dict, ref_fp, out_dir, to_delete, fsldir, quiet=True):
    """
    Register a set of magnitude NIfTI files to a reference with FSL flirt
    (rigid-body 6 degrees of freedom, mutual information cost-function)

    Apply the relevant transformation matrix to phase images

    :param fp_dict: dictionary of NIfTI files
    :type fp_dict: dict
    :param ref_fp: reference image filepath
    :type ref_fp: pathlib.Path
    :param out_dir: output directory
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :type to_delete: list
    :param fsldir: FSL directory
    :type fsldir: pathlib.Path
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: fp_dict, to_delete
    :rtype: tuple
    """

    # Register e1 image to reference
    base_fn = remove_file_ext(fp_dict["e1_fp"]).name
    e1_r_fn = base_fn + "_r.nii.gz"
    e1_r_fp = out_dir / e1_r_fn
    e1_omat_fn = base_fn + "_xform.mat"
    e1_omat_fp = out_dir / e1_omat_fn

    flirt_cmd = [
        str(fsldir / "bin" / "flirt"),
        "-cost",
        "mutualinfo",
        "-2D",
        "-in",
        str(fp_dict["e1_fp"]),
        "-ref",
        str(ref_fp),
        "-out",
        str(e1_r_fp),
        "-omat",
        str(e1_omat_fp),
    ]

    if not quiet:
        print("***", " ".join(flirt_cmd))

    # Capture the output from flirt as user doesn't need to see it
    sp.run(flirt_cmd, check=True, text=True, capture_output=True)

    # Register e2 image to reference (by applying xform from e1)
    base_fn = remove_file_ext(fp_dict["e2_fp"]).name
    e2_r_fn = base_fn + "_r.nii.gz"
    e2_r_fp = out_dir / e2_r_fn

    flirt_cmd = [
        str(fsldir / "bin" / "flirt"),
        "-in",
        str(fp_dict["e2_fp"]),
        "-ref",
        str(ref_fp),
        "-out",
        str(e2_r_fp),
        "-applyxfm",
        "-init",
        str(e1_omat_fp),
    ]

    if not quiet:
        print("***", " ".join(flirt_cmd))

    # Capture the output from flirt as user doesn't need to see it
    sp.run(flirt_cmd, check=True, text=True, capture_output=True)

    fp_dict["e1_fp"] = e1_r_fp
    fp_dict["e2_fp"] = e2_r_fp

    to_delete.extend([e1_r_fp, e2_r_fp])

    return fp_dict, to_delete


def mask(fp_dict, mask_fp, out_dir, to_delete, fsldir, quiet=True):

    """
    Apply mask using fslmaths to a set of NIfTI files

    :param fp_dict: dictionary of NIfTI files
    :type fp_dict: dict
    :param mask_fp: filepath of mask image
    :type mask_fp: pathlib.Path
    :param out_dir: output directory
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :type to_delete: list
    :param fsldir: FSL directory
    :type fsldir: pathlib.Path
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: fp_dict, to_delete
    :rtype: tuple
    """

    for key, fp in fp_dict.items():
        out_fn = remove_file_ext(fp).name + "_m.nii.gz"
        out_fp = out_dir / out_fn
        # Keep a list of intermediate files to potentially delete
        to_delete.append(out_fp)

        fslmaths_cmd = [
            str(fsldir / "bin" / "fslmaths"),
            str(fp),
            "-mas",
            str(mask_fp),
            str(out_fp),
        ]

        if not quiet:
            print("***", " ".join(fslmaths_cmd))

        sp.run(fslmaths_cmd, check=True, text=True)

        fp_dict[key] = out_fp

    return fp_dict, to_delete


def crop(fp_dict, crop_dims, out_dir, to_delete, fsldir, quiet=True):

    """
    Crop a set of NIfTI files using fslroi

    :param fp_dict: dictionary of NIfTI files
    :type fp_dict: dict
    :param crop_dims: list of start coordinates and sizes for cropping
    :type crop_dims: list
    :param out_dir: output directory
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :type to_delete: list
    :param fsldir: FSL directory
    :type fsldir: pathlib.Path
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: fp_dict, to_delete
    :rtype: tuple
    """

    for key, fp in fp_dict.items():
        out_fn = remove_file_ext(fp).name + "_c.nii.gz"
        out_fp = out_dir / out_fn
        to_delete.append(out_fp)

        fslroi_cmd = [
            str(fsldir / "bin" / "fslroi"),
            str(fp),
            str(out_fp),
            str(crop_dims[0]),
            str(crop_dims[1]),
            str(crop_dims[2]),
            str(crop_dims[3]),
            str(crop_dims[4]),
            str(crop_dims[5]),
        ]

        if not quiet:
            print("***", " ".join(fslroi_cmd))

        sp.run(fslroi_cmd, check=True, text=True)

        fp_dict[key] = out_fp

    return fp_dict, to_delete


def resample(fp_dict, ref_fp, out_dir, to_delete, fsldir, quiet=True):

    """
    Resample set of NIfTI files to reference space with FSL flirt

    :param fp_dict: dictionary of NIfTI files
    :type fp_dict: dict
    :param ref_fp: reference NIfTI filepath
    :type ref_fp: pathlib.Path
    :param out_dir: output directory
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :type to_delete: list
    :param fsldir: FSL directory
    :type fsldir: pathlib.Path
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: fp_dict, delete_list
    :rtype: (dict, list)
    """

    unity_xform_fp = out_dir / "unity.mat"
    unity_xform_fp.write_text("1 0 0 0 \n0 1 0 0 \n0 0 1 0 \n0 0 0 1")

    for key, fp in fp_dict.items():
        out_fn = remove_file_ext(fp).name + "_resamp.nii.gz"
        out_fp = out_dir / out_fn

        flirt_cmd = [
            str(fsldir / "bin" / "flirt"),
            "-in",
            str(fp),
            "-ref",
            str(ref_fp),
            "-out",
            str(out_fp),
            "-applyxfm",
            "-init",
            str(unity_xform_fp),
        ]

        if not quiet:
            print("***", " ".join(flirt_cmd))

        # Capture the output from flirt as user doesn't need to see it
        sp.run(flirt_cmd, check=True, text=True, capture_output=True)

        fp_dict[key] = out_fp

    to_delete.extend([unity_xform_fp, out_fp])

    return fp_dict, to_delete


def create_mask(fp, out_dir, to_delete, fsldir, quiet=True):
    """
    Create a mask using fslmaths thresholding, binarising and eroding

    :param fp:
    :type fp: pathlib.Path
    :param out_dir: directory to store MTR map NIfTI file in
    :type out_dir: pathlib.Path
    :param to_delete: intermediate files to delete
    :type to_delete: list
    :param fsldir: full path to FSL directory
    :type fsldir: pathlib.Path
    :param quiet: don't display information messages or progress status
    :type quiet: bool
    :return: bgmask_fp, to_delete
    :rtype: tuple(str, list)
    """

    bg_mask_fp = out_dir / "bgmask.nii.gz"

    fslmaths_cmd = [
        str(fsldir / "bin" / "fslmaths"),
        str(fp),
        "-thr",
        "5",
        "-bin",
        "-kernel",
        "2D",
        "-ero",
        str(bg_mask_fp),
    ]

    if not quiet:
        print("****", " ".join(fslmaths_cmd))

    sp.run(fslmaths_cmd, check=True, text=True)

    to_delete.extend(
        [
            bg_mask_fp,
        ]
    )

    return bg_mask_fp, to_delete

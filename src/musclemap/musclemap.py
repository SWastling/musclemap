"""
Calculate B1, fat fraction, magnetisation transfer ratio or T2 maps
"""

import argparse
import pathlib
import subprocess as sp
import sys

import importlib.metadata

import musclemap.b1 as b1
import musclemap.config as config
import musclemap.ff as ff
import musclemap.mtr as mtr
import musclemap.mtr_b1 as mtr_b1
import musclemap.preproc as preproc
import musclemap.t2 as t2
import musclemap.vercheck as vercheck

__version__ = importlib.metadata.version("musclemap")


def main():
    parser = argparse.ArgumentParser(
        description="calculate B1, fat fraction, "
        "magnetisation transfer "
        "ratio or T2 maps"
    )

    parser.add_argument(
        "-o",
        help="output folder for results (defaults are "
        "./b1, ./ff, ./mtr, ./mtr-b1 "
        " or ./t2",
        metavar="output_folder",
        type=pathlib.Path,
    )

    parser.add_argument(
        "-r", help="register images to first " "source file", action="store_true"
    )

    parser.add_argument("-m", help="filename of mask images", metavar="mask")

    parser.add_argument(
        "-c",
        help="crop images (indexing starts with 0 not 1,"
        "inputting -1 for a size will set it to the "
        "full image extent for that dimension)",
        type=int,
        nargs=6,
        metavar=("xmin", "xsize", "ymin", "ysize", "zmin", "zsize"),
    )

    parser.add_argument("-k", help="keep intermediate images", action="store_true")

    parser.add_argument("-v", help="view results in FSLeyes", action="store_true")

    parser.add_argument(
        "-quiet",
        help="don't display information messages or " "progress status",
        action="store_true",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--any-version",
        dest="any_version",
        default=False,
        action="store_true",
        help="don't abort if version checks fail",
    )

    subparsers = parser.add_subparsers(
        title="algorithm", dest="algorithm", required=True
    )

    parser_b1 = subparsers.add_parser(
        "b1",
        help="calculate B1 map from "
        "double-angle (60 and 120 "
        "degree) spin echo data",
        description="calculate B1 map from "
        "double-angle (60 and 120 "
        "degree) spin echo data",
    )

    parser_ff = subparsers.add_parser(
        "ff",
        help="ff help",
        description="calculate fat fraction maps from three-point Dixon data",
    )

    parser_mtr = subparsers.add_parser(
        "mtr",
        help="mtr help",
        description="calculate magnetisation transfer "
        "ratio (MTR) map from MT on and off "
        "data",
    )

    parser_mtr_b1 = subparsers.add_parser(
        "mtr-b1",
        help="mtr-b1 help",
        description="calculate magnetisation transfer "
        "ratio (MTR) map from MT on and off "
        "data, with correction for B1 "
        "inhomogeneities as described in "
        "Sinclair et al. NMR in Biomedicine "
        "2012",
    )

    parser_t2 = subparsers.add_parser(
        "t2", help="calculate T2 map from double-" "echo spin-echo data"
    )

    parser_b1.add_argument(
        "fa60", help="60 degree flip-angle spin-echo image", type=pathlib.Path
    )

    parser_b1.add_argument(
        "fa120", help="120 degree flip-angle spin-echo " "image", type=pathlib.Path
    )

    parser_ff.add_argument(
        "m_1",
        help="out-of-phase fat-water magnitude image" " (e.g. TE 3.45 ms at 3 T)",
        type=pathlib.Path,
    )

    parser_ff.add_argument(
        "phi_1",
        help="out-of-phase fat-water phase image " "(e.g. TE 3.45 ms at 3 T)",
        type=pathlib.Path,
    )

    parser_ff.add_argument(
        "m0",
        help="in-phase fat-water magnitude image " "(e.g. TE 4.60 ms at 3 T)",
        type=pathlib.Path,
    )

    parser_ff.add_argument(
        "phi0",
        help="in-phase fat-water phase image " "(e.g. TE 4.60 ms at 3 T)",
        type=pathlib.Path,
    )

    parser_ff.add_argument(
        "m1",
        help="out-of-phase fat-water magnitude image " "(e.g. TE 5.75 ms at 3 T)",
        type=pathlib.Path,
    )

    parser_ff.add_argument(
        "phi1",
        help="out-of-phase fat-water phase image " "(e.g. TE 5.75 ms at 3 T)",
        type=pathlib.Path,
    )

    parser_ff.add_argument(
        "scanner",
        metavar="scanner",
        type=str,
        choices=["ge", "siemens", "philips"],
        help="scanner type, chose from {%(choices)s}",
    )

    parser_ff.add_argument(
        "-s",
        help="separate images in x-direction during"
        " phase unwrapping (recommended for "
        "legs)",
        action="store_true",
    )

    parser_ff.add_argument(
        "-nb",
        help="calculate noise-bias corrected "
        "fat-fraction map - see Liu et al. "
        "MRM 2007",
        action="store_true",
    )

    parser_mtr.add_argument("mt_on", help="MT on magnitude image", type=pathlib.Path)

    parser_mtr.add_argument("mt_off", help="MT off magnitude image", type=pathlib.Path)

    parser_mtr_b1.add_argument("mt_on", help="MT on magnitude image", type=pathlib.Path)

    parser_mtr_b1.add_argument(
        "mt_off", help="MT off magnitude image", type=pathlib.Path
    )

    parser_mtr_b1.add_argument(
        "fa60",
        help="60 degree flip-angle image (for " "B1 inhomogeneity correction)",
        type=pathlib.Path,
    )

    parser_mtr_b1.add_argument(
        "fa120",
        help="120 degree flip-angle image (for " "B1 inhomogeneity correction)",
        type=pathlib.Path,
    )

    parser_mtr_b1.add_argument(
        "ref",
        help="reference image used for masking "
        "and resampling typically use "
        "out-of-phase fat-water magnitude "
        "image (e.g. TE 3.45 ms at 3 T) "
        "Dixon",
        type=pathlib.Path,
    )

    parser_mtr_b1.add_argument(
        "-b1pcf",
        default=0.0085,
        help="population level B1 correction factor. "
        "This is the gradient of a straight line "
        "fit of B1 to MTR (default %(default)s)",
        type=float,
        metavar="b1pcf",
    )

    parser_t2.add_argument("e1", help="echo 1 image", type=pathlib.Path)

    parser_t2.add_argument("e2", help="echo 2 image", type=pathlib.Path)

    parser_t2.add_argument("te1", help="TE of echo 1 in ms (e.g. 16)", type=float)

    parser_t2.add_argument("te2", help="TE of echo 2 in ms (e.g. 56)", type=float)

    parser_t2.add_argument(
        "ref",
        help="reference image used for registration"
        " typically use out-of-phase "
        "fat-water magnitude image (e.g. "
        "TE 3.45 ms at 3 T) Dixon",
        type=pathlib.Path,
    )

    if len(sys.argv) == 1:
        sys.argv.append("-h")

    args = parser.parse_args()

    print("* checking version of FSL")
    fsldir = vercheck.get_fsldir()
    vercheck.check_lib_ver(
        "FSL", vercheck.get_fsl_ver(fsldir), config.FSL_VERSIONS, args.any_version
    )

    if args.o:
        out_dir = args.o.resolve()
    else:
        out_dir = pathlib.Path(args.algorithm).resolve()

    if args.algorithm == "b1":
        fp_dict = {"fa60_fp": args.fa60, "fa120_fp": args.fa120}

        ref_fp = fp_dict["fa60_fp"]

        dr = [0.0, 2.0]

    elif args.algorithm == "ff":

        fp_dict = {
            "mminus1_fp": args.m_1,
            "phiminus1_fp": args.phi_1,
            "m0_fp": args.m0,
            "phi0_fp": args.phi0,
            "m1_fp": args.m1,
            "phi1_fp": args.phi1,
        }

        ref_fp = fp_dict["mminus1_fp"]

        dr = [0.0, 100.0]

    elif args.algorithm == "mtr":

        fp_dict = {"mt_on_fp": args.mt_on, "mt_off_fp": args.mt_off}

        ref_fp = fp_dict["mt_on_fp"]

        dr = [0.0, 100.0]

    elif args.algorithm == "mtr-b1":

        fp_dict = {"mt_on_fp": args.mt_on, "mt_off_fp": args.mt_off}

        ref_fp = fp_dict["mt_on_fp"]

        fp_check_dict = {
            "mt_on_fp": args.mt_on,
            "mt_off_fp": args.mt_off,
            "fa60_fp": args.fa60,
            "fa120_fp": args.fa120,
            "res_ref_fp": args.ref,
        }

        if any((args.r, args.c, args.m)):
            sys.stderr.write(
                "* optional arguments -r, -c or -m cannot be used"
                " with mtr-b1 algorithm - exiting \n"
            )
            sys.exit(1)

        dr = [0.0, 100.0]

    else:
        # t2
        fp_dict = {"e1_fp": args.e1, "e2_fp": args.e2}

        ref_fp = fp_dict["e1_fp"]

        fp_check_dict = {"e1_fp": args.e1, "e2_fp": args.e2, "res_ref_fp": args.ref}

        if any((args.r, args.c, args.m)):
            sys.stderr.write(
                "* optional arguments -r, -c or -m cannot be used"
                " with t2 algorithm \n"
            )
            sys.exit(1)

        dr = [0.0, 200.0]

    # Check if the input files exist
    if not args.quiet:
        print("* performing pre-processing steps")
        print("** checking NIfTI files exist")

    if args.algorithm == "mtr-b1" or args.algorithm == "t2":
        preproc.check_files_exist(fp_check_dict)
    else:
        preproc.check_files_exist(fp_dict)

    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    # Save the command run by the user
    command_fp = out_dir / "processing_command.txt"
    command_fp.write_text("%s" % " ".join(sys.argv))

    # Generate a list of intermediate files to delete unless instructed by user
    to_delete = []

    if not args.quiet:
        print("** checking dimensions and orientations are identical")
    preproc.check_shape_and_orientation(fp_dict, ref_fp)

    # Pre-processing - register, crop and mask

    if args.algorithm == "ff":
        fp_dict, to_delete = preproc.unscale(fp_dict, out_dir, to_delete)

    if args.r:
        if not args.quiet:
            print("** registering images to %s with fsl flirt" % ref_fp)
        if args.algorithm == "ff":
            fp_dict, to_delete = preproc.register_dixon(
                fp_dict, out_dir, to_delete, fsldir, args.quiet
            )
        else:
            fp_dict, to_delete = preproc.register(
                fp_dict, ref_fp, out_dir, to_delete, fsldir, args.quiet
            )

    if args.m:
        if not args.quiet:
            print("** masking with fslmaths")

        fp_dict, to_delete = preproc.mask(
            fp_dict, args.m, out_dir, to_delete, fsldir, args.quiet
        )

    if args.c:
        if not args.quiet:
            print("** cropping with fslroi")

        fp_dict, to_delete = preproc.crop(
            fp_dict, args.c, out_dir, to_delete, fsldir, args.quiet
        )

    # Calculate appropriate map depending on algorithm chosen
    if args.algorithm == "b1":
        if not args.quiet:
            print("* calculating B1 map")

        map_fp = b1.process_b1(fp_dict, out_dir, args.quiet)

    elif args.algorithm == "ff":
        if not args.quiet:
            print("* calculating fat-fraction map(s)")

        map_fp, to_delete = ff.process_ff(
            fp_dict,
            out_dir,
            to_delete,
            fsldir,
            args.nb,
            args.scanner,
            args.s,
            args.quiet,
        )
    elif args.algorithm == "mtr":
        if not args.quiet:
            print("* calculating MTR map")

        map_fp = mtr.process_mtr(fp_dict, out_dir, args.quiet)

    elif args.algorithm == "mtr-b1":
        if not args.quiet:
            print("* calculating MTR map with B1 correction")

        map_fp, to_delete = mtr_b1.process_mtr_b1(
            fp_dict,
            args.fa60,
            args.fa120,
            args.ref,
            out_dir,
            to_delete,
            fsldir,
            args.b1pcf,
            args.quiet,
        )
    else:
        # t2
        if not args.quiet:
            print("* calculating T2 map")
        map_fp, delete_list = t2.process_t2(
            fp_dict,
            args.te1,
            args.te2,
            args.ref,
            out_dir,
            to_delete,
            fsldir,
            args.quiet,
        )

    if not args.quiet:
        print("* performing post-processing steps")

    if not args.k:
        if not args.quiet:
            print("** deleting unwanted intermediate NIfTI files")

        # Remove non-unique entries in the list by converting to a set and back
        to_delete = list(set(to_delete))

        for item in to_delete:
            item.unlink()

    if args.v:
        if not args.quiet:
            print("** opening map(s) with fsleyes")

        fsleyes_cmd = [str(fsldir / "bin" / "fsleyes")]
        if isinstance(map_fp, list):
            for item in map_fp:
                fsleyes_cmd.extend([str(item), "-dr", str(dr[0]), str(dr[1])])
        else:
            fsleyes_cmd.extend([str(map_fp), "-dr", str(dr[0]), str(dr[1])])

        if not args.quiet:
            print("***", " ".join(fsleyes_cmd))

        sp.Popen(fsleyes_cmd, stderr=sp.PIPE, stdout=sp.PIPE)


if __name__ == "__main__":  # pragma: no cover
    main()

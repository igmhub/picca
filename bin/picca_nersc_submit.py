#!/usr/bin/env python
"""Writes scripts to send the picca BAO analysis

Once ran, run `./submit.sh` in your terminal. This script works on nersc::cori
not on nersc::edison.
"""
import sys
import os
from os.path import basename, dirname, stat
import argparse


class Batch:
    """
    Generic class to hold the names of the .batch slurm submission files.

    Attributes:
        out_dir: string or  None
            Output directory
        picca_deltas_script_filename: string or None
            .batch file to submit the picca_deltas.py script
        cf_script_list: list of strings
            List of .batch files to submit the picca_cf.py script
        export_script_list: list of strings
            List of .batch files to submit the picca_export.py and
            picca_fitter2.py scripts for the auto-correlation analysis
        dmat_script_list: list of strings
            List of .batch files to submit the picca_dmat.py script
        xcf_script_list: list of strings
            List of .batch files to submit the picca_xcf.py script
        xdmat_script_list: list of strings
            List of .batch files to submit the picca_xdmat.py script
        xexport_script_list: list of strings
            List of .batch files to submit the picca_export.py and
            picca_fitter2.py scripts for the cross-correlation analysis
    """

    def __init__(self):
        """Initialize class instance"""
        self.out_dir = None
        self.picca_deltas_script_filename = None
        self.cf_script = []
        self.export_script = []
        self.dmat_script = []
        self.xcf_script = []
        self.xdmat_script = []
        self.xexport_script = []


def get_header(time, name, email=None, queue="regular", account="desi"):
    """Returns the standard header for running the analyses.

    Args:
        time: string
            Requested maximum runtime, format: hh:mm:ss
        name: string
            Name of the job
        email: string or None - default: None
            Email address to send progress to
        queue: string - default: "regular"
            Name of the submission queue
        account: string - default: "desi"
            Name of the submission account

    Returns:
        A string holding the header
    """
    header = ""
    header += "#!/bin/bash\n"
    header += "#SBATCH -N 1\n"
    header += "#SBATCH -C haswell\n"
    header += "#SBATCH -q {}\n".format(queue)
    header += "#SBATCH -J {}\n".format(name)
    if email is not None:
        header += "#SBATCH --mail-user={}\n".format(email)
        header += "#SBATCH --mail-type=ALL\n"
    header += "#SBATCH -t {}\n".format(time)
    header += "#SBATCH -L project\n"
    header += "#SBATCH -A {}\n".format(account)
    header += "#OpenMP settings:\n"
    header += "export OMP_NUM_THREADS=1\n"

    return header


def picca_deltas_script(batch,
                        time,
                        in_dir,
                        out_dir,
                        drq_filename,
                        email=None,
                        debug=False,
                        mode="desi",
                        lambda_rest_frame_min=None,
                        lambda_rest_frame_max=None):
    """Writes a .batch file to submit the picca_deltas.py script

    Args:
        batch: Batch
            a Batch instance
        time: string
            Requested maximum runtime, format: hh:mm:ss
        in_dir: string
            Input directory
        out_dir: string
            Output directory
        drq_filename: string
            Absolute path for the DRQ catalogue
        email: string or None - default: None
            Email address to send progress to
        debug: bool - default: False
            If True, run only over 10000 quasars (and on the debug queue)
        mode: "desi" or "eboss" - default: "desi"
            Open mode of the spectra files. "eboss" will load the script using
            the mode "spplate"
        lambda_rest_frame_min: float or None - default: None
            Minimum rest-frame wavelengthe (in Angs)
        lambda_rest_frame_max: float or None - default: None
            Maximum rest-frame wavelengthe (in Angs)
    """
    assert mode in ["eboss", "desi"]
    if mode == "eboss":
        mode = "spplate"
    header = get_header(time, name="picca_deltas", email=email)
    header += ("srun -n 1 -c 64 picca_deltas.py " +
               "--in-dir {} --drq {} ".format(in_dir, drq_filename) +
               "--out-dir {}/deltas/ --mode {} ".format(out_dir, mode) +
               "--iter-out-prefix {}/iter ".format(out_dir) +
               "--log {}/input.log".format(out_dir))
    if lambda_rest_frame_min is not None:
        header += " --lambda-rest-min {}".format(lambda_rest_frame_min)
    if lambda_rest_frame_max is not None:
        header += " --lambda-rest-max {}".format(lambda_rest_frame_max)
    if debug:
        header += " --nspec 10000"

    header += "\n"
    batch.picca_deltas_script_filename = "picca_deltas.batch"
    file = open(out_dir + "/picca_deltas.batch", "w")
    file.write(header)
    file.close()


def cf_script(batch,
              time,
              z_intervals,
              out_dir,
              email=None,
              fid_om=None,
              fid_pk=None,
              fid_or=None):
    """
    Writes the .batch files to submit the picca_cf.py and the picca_export.py
    scripts and adds them to the batch.cf and batch.export lists

    Args:
        batch: Batch
            a Batch instance
        time: string
            Requested maximum runtime, format: hh:mm:ss
        z_intervals: list of strings
            A list holding z intervals. Each interval is of the format
            "z_min:z_max"
        out_dir: string
            Output directory
        email: string or None - default: None
            Email address to send progress to
        fid_om: float or None -  default: None
            Matter density for the fiducial cosmology
        fid_Pk: string or None - default: None
            Fiducial P(k)
        fid_or: float or None - default: None
            Radiation density for the fiducial cosmology
    """
    for z_interval in z_intervals:
        z_min, z_max = z_interval.split(":")
        out = "cf_z_{}_{}.fits".format(z_min, z_max)
        exp_batch = export_script(
            "00:10:00",
            out_dir + "/cf_z_{}_{}.fits".format(z_min, z_max),
            out_dir + "/dmat_z_{}_{}.fits".format(z_min, z_max),
            out_dir + "/cf_z_{}_{}-exp.fits".format(z_min, z_max),
            fid_pk=fid_pk)
        header = get_header(time, name=out, email=email)
        srun = (header + "srun -n 1 -c 64 "
                "picca_cf.py --in-dir {}/deltas/ ".format(out_dir) +
                "--z-cut-min {} --z-cut-max {} ".format(z_min, z_max) +
                "--out {}/{} --nproc 32 ".format(out_dir, out) +
                "--fid-Om {} --fid-Or {} \n".format(fid_om, fid_or))

        batch_filename = out_dir + "/" + out.replace(".fits", ".batch")
        batch.cf_script_list.append(basename(batch_filename))
        batch.export_script_list.append(basename(exp_batch))

        file = open(batch_filename, "w")
        file.write(srun)
        file.close()


def dmat_script(batch,
                time,
                z_intervals,
                out_dir,
                email=None,
                reject=0.99,
                fid_om=None,
                fid_or=None):
    """
    Writes the .batch files to submit the picca_dmat.py script
    and adds them to the batch.dmat script
    Inputs:
        batch: Batch
            a Batch instance
        time: string
            Requested maximum runtime, format: hh:mm:ss
        z_intervals: list of strings
            A list holding z intervals. Each interval is of the format
            "z_min:z_max"
        out_dir: string
            Output directory
        email: string or None - default: None
            Email address to send progress to
        reject: float - default: 0.99
            Fraction of pairs to be rejected
        fid_om: float or None -  default: None
            Matter density for the fiducial cosmology
        fid_or: float or None - default: None
            Radiation density for the fiducial cosmology
    """
    for z_interval in z_intervals:
        z_min, z_max = z_interval.split(":")
        out = "dmat_z_{}_{}.fits".format(z_min, z_max)
        header = get_header(time, name=out, email=email)
        srun = (header + "srun -n 1 -c 64 " +
                "picca_dmat.py --in-dir {}/deltas/ ".format(out_dir) +
                "--z-cut-min {} --z-cut-max {} ".format(z_min, z_max) +
                "--out {}/{} --rej {} ".format(out_dir, out, reject) +
                "--nproc 32  --fid-Om {} --fid-Or {}\n".format(fid_om, fid_or))
        batch_filename = out_dir + "/" + out.replace(".fits", ".batch")
        batch.dmat_script_list.append(basename(batch_filename))

        file = open(batch_filename, "w")
        file.write(srun)
        file.close()


def xcf_script(batch,
               time,
               drq_filename,
               z_intervals,
               out_dir,
               email=None,
               fid_om=None,
               fid_pk=None,
               fid_or=None):
    """
    Writes the .batch files to submit the picca_xcf.py script
    and the picca_export.py scripts
    and adds them to the b.xcf and b.xexport lists

    Args:
        batch: Batch
            a Batch instance
        time: string
            Requested maximum runtime, format: hh:mm:ss
        drq_filename: string
            Absolute path for the DRQ catalogue
        z_intervals: list of strings
            A list holding z intervals. Each interval is of the format
            "z_min:z_max"
        out_dir: string
            Output directory
        email: string or None - default: None
            Email address to send progress to
        fid_om: float or None -  default: None
            Matter density for the fiducial cosmology
        fid_Pk: string or None - default: None
            Fiducial P(k)
        fid_or: float or None - default: None
            Radiation density for the fiducial cosmology
    """
    for z_interval in z_intervals:
        z_min, z_max = z_interval.split(":")
        out = "xcf_z_{}_{}.fits".format(z_min, z_max)
        header = get_header(time, name=out, email=email)
        exp_batch = export_script(
            "00:10:00",
            out_dir + "/xcf_z_{}_{}.fits".format(z_min, z_max),
            out_dir + "/xdmat_z_{}_{}.fits".format(z_min, z_max),
            out_dir + "/xcf_z_{}_{}-exp.fits".format(z_min, z_max),
            fid_pk=fid_pk)
        srun = (header + "srun -n 1 -c 64 picca_xcf.py " +
                "--drq {} --in-dir {}/deltas/ ".format(drq_filename, out_dir) +
                "--z-evol-obj 1.44 " +
                "--z-cut-min {} --z-cut-max {} ".format(z_min, z_max) +
                "--out {}/{} --nproc 32 ".format(out_dir, out) +
                "--fid-Om {} --fid-Or {}\n".format(fid_om, fid_or))
        batch_filename = out_dir + "/" + out.replace(".fits", ".batch")
        batch.xcf_script_list.append(basename(batch_filename))
        batch.xexport_script_list.append(basename(exp_batch))

        file = open(batch_filename, "w")
        file.write(srun)
        file.close()


def xdmat_script(batch,
                 time,
                 drq_filename,
                 z_intervals,
                 out_dir,
                 email=None,
                 reject=0.95,
                 fid_om=None,
                 fid_or=None):
    """
    Writes the .batch files to submit the picca_xdmat.py script
    and adds if to the b.xdmat list

    Args:
        batch: Batch
            a Batch instance
        time: string
            Requested maximum runtime, format: hh:mm:ss
        drq_filename: string
            Absolute path for the DRQ catalogue
        z_intervals: list of strings
            A list holding z intervals. Each interval is of the format
            "z_min:z_max"
        out_dir: string
            Output directory
        email: string or None - default: None
            Email address to send progress to
        reject: float - default: 0.99
            Fraction of pairs to be rejected
        fid_om: float or None -  default: None
            Matter density for the fiducial cosmology
        fid_or: float or None - default: None
            Radiation density for the fiducial cosmology
    """
    for z_interval in z_intervals:
        z_min, z_max = z_interval.split(":")
        out = "xdmat_z_{}_{}.fits".format(z_min, z_max)
        header = get_header(time, name=out, email=email)
        srun = (header + "srun -n 1 -c 64 picca_xdmat.py " +
                "--drq {} --in-dir {}/deltas/ ".format(drq_filename, out_dir) +
                "--z-evol-obj 1.44 " +
                "--z-cut-min {} --z-cut-max {} ".format(z_min, z_max) +
                "--out {}/{} ".format(out_dir, out) +
                "--rej {} --nproc 32 ".format(reject) +
                "--fid-Om {} --fid-Or {}\n".format(fid_om, fid_or))
        batch_filename = out_dir + "/" + out.replace(".fits", ".batch")
        batch.xdmat_script_list.append(basename(batch_filename))

        file = open(batch_filename, "w")
        file.write(srun)
        file.close()


def export_script(time, cf_file, dmat_file, out, fid_pk):
    """Writes the .batch file to submit the picca_export.py and picca_fitter2.py
    scripts

    Args:
        time: string
            Requested maximum runtime, format: hh:mm:ss
        cf_file: string
            Path to the file containing the (auto) correlation function
        dmat_file: string
            Path to the file containing the distortion matrix for the
            autocorrelation
        out: string
            Output of the script picca_export.py
        fid_Pk: string
            Fiducial P(k)

    Returns:
        Filename of the batch file to submit the picca_export.py and
        picca_fitter2.py
    """
    header = get_header(time, name=basename(out), queue="regular")
    srun = (header + "srun -n 1 -c 64 picca_export.py " +
            "--data {} --dmat {} ".format(cf_file, dmat_file) +
            "--out {}\n".format(out))
    chi2_ini = do_ini(dirname(out), basename(out), fid_pk)
    srun += "srun -n 1 -c 64 picca_fitter2.py {}\n".format(chi2_ini)
    batch_filename = out.replace(".fits", ".batch")
    file = open(batch_filename, "w")
    file.write(srun)
    file.close()

    return batch_filename


def do_ini(out_dir, cf_file, fid_pk):
    """Writes the .ini files to control the fits

    Args:
        out: string
            Output of the script picca_export.py
        cf_file: string
            path to the input file for the fitter
        fid_Pk: string
            Fiducial P(k)

    Returns:
        The (basename) of the chi2.ini file
    """
    file = open(out_dir + "/" + cf_file.replace(".fits", ".ini"), "w")
    file.write("[data]\n")
    file.write("name = {}\n".format(basename(cf_file).replace(".fits", "")))
    file.write("tracer1 = LYA\n")
    file.write("tracer1-type = continuous\n")
    if "xcf" in cf_file:
        file.write("tracer2 = QSO\n")
        file.write("tracer2-type = discrete\n")
    else:
        file.write("tracer2 = LYA\n")
        file.write("tracer2-type = continuous\n")
    file.write("filename = {}\n".format(out_dir + '/' + cf_file))
    file.write("ell-max = 6\n")

    file.write("[cuts]\n")
    file.write("rp-min = -200\n")
    file.write("rp-max = 200\n")

    file.write("rt-min = 0\n")
    file.write("rt-max = 200\n")

    file.write("r-min = 10\n")
    file.write("r-max = 180\n")

    file.write("mu-min = -1\n")
    file.write("mu-max = 1\n")

    file.write("[model]\n")
    file.write("model-pk = pk_kaiser\n")
    file.write("z evol LYA = bias_vs_z_std\n")
    file.write("growth function = growth_factor_de\n")
    if "xcf" in cf_file:
        file.write("model-xi = xi_drp\n")
        file.write("z evol QSO = bias_vs_z_std\n")
        file.write("velocity dispersion = pk_velo_lorentz\n")
    else:
        file.write("model-xi = xi\n")

    file.write("[parameters]\n")

    file.write("ap = 1. 0.1 0.5 1.5 free\n")
    file.write("at = 1. 0.1 0.5 1.5 free\n")
    file.write("bias_eta_LYA = -0.17 0.017 None None free\n")
    file.write("beta_LYA = 1. 0.1 None None free\n")
    file.write("alpha_LYA = 2.9 0.1 None None fixed\n")
    if "xcf" in cf_file:
        file.write("bias_eta_QSO = 1. 0.1 None None fixed\n")
        file.write("beta_QSO = 0.3 0.1 None None fixed\n")
        file.write("alpha_QSO = 1.44 0.1 None None fixed\n")

    file.write("growth_rate = 0.962524 0.1 None None fixed\n")

    file.write("sigmaNL_par = 6.36984 0.1 None None fixed\n")
    file.write("sigmaNL_per = 3.24 0.1 None None fixed\n")

    file.write(("par binsize {} = 4. 0.4 None None "
                "free\n").format(cf_file.replace(".fits", "")))
    file.write(("per binsize {} = 4. 0.4 None None "
                "free\n").format(cf_file.replace(".fits", "")))

    file.write("bao_amp = 1. 0.1 None None fixed\n")
    if "xcf" in cf_file:
        file.write("drp_QSO = 0. 0.1 None None free\n")
        file.write("sigma_velo_lorentz_QSO = 5. 0.1 None None free\n")
    file.close()

    chi2_ini = out_dir + "/chi2_{}".format(cf_file.replace(".fits", ".ini"))
    file = open(chi2_ini, "w")
    file.write("[data sets]\n")
    file.write("zeff = 2.310\n")
    file.write("ini files = {}\n".format(out_dir + "/" +
                                         cf_file.replace(".fits", ".ini")))

    file.write("[fiducial]\n")
    file.write("filename = {}\n".format(fid_pk))

    file.write("[verbosity]\n")
    file.write("level = 0\n")

    file.write("[output]\n")
    file.write("filename = {}\n".format(out_dir + "/" +
                                        cf_file.replace(".fits", ".h5")))

    file.write("[cosmo-fit type]\n")
    file.write("cosmo fit func = ap_at\n")
    file.close()

    return chi2_ini


def submit(batch):
    """Writes the .batch file to submit all the scripts created previously

    Args:
        batch: Batch
            a Batch instance
    """
    out_name = batch.out_dir + "/submit.sh"
    file = open(out_name, "w")
    file.write("#!/bin/bash\n")
    if batch.picca_deltas_script_filename is not None:
        file.write(("picca_deltas=$(sbatch --parsable "
                    "{})\n").format(batch.picca_deltas_script_filename))
        file.write('echo "picca_deltas: "$picca_deltas\n')
    for cf_batch, dmat_batch, exp_batch in zip(batch.cf_script_list,
                                               batch.dmat_script_list,
                                               batch.export_script_list):
        var_cf = cf_batch.replace(".batch", "").replace(".", "_")
        var_dmat = dmat_batch.replace(".batch", "").replace(".", "_")
        if batch.picca_deltas_script_filename is not None:
            file.write(("{}=$(sbatch --parsable "
                        "--dependency=afterok:$picca_deltas "
                        "{})\n").format(var_cf, cf_batch))
            file.write('echo "{0}: "${0} \n'.format(var_cf))
            file.write(("{}=$(sbatch --parsable "
                        "--dependency=afterok:$picca_deltas "
                        "{})\n").format(var_dmat, dmat_batch))
            file.write('echo "{0}: "${0} \n'.format(var_dmat))
        else:
            file.write("{}=$(sbatch --parsable {})\n".format(var_cf, cf_batch))
            file.write('echo "{0}: "${0} \n'.format(var_cf))
            file.write("{}=$(sbatch --parsable {})\n".format(
                var_dmat, dmat_batch))
            file.write('echo "{0}: "${0} \n'.format(var_dmat))
        var_exp = exp_batch.replace(".batch",
                                    "").replace(".", "_").replace("-", "_")
        file.write(("{}=$(sbatch --parsable "
                    "--dependency=afterok:${},afterok:${} "
                    "{})\n").format(var_exp, var_cf, var_dmat, exp_batch))
        file.write('echo "{0}: "${0} \n'.format(var_exp))

    for xcf_batch, xdmat_batch, xexp_batch in zip(batch.xcf_script_list,
                                                  batch.xdmat_script_list,
                                                  batch.xexport_script_list):
        var_xcf = xcf_batch.replace(".batch", "").replace(".", "_")
        var_xdmat = xdmat_batch.replace(".batch", "").replace(".", "_")
        if batch.picca_deltas_script_filename is not None:
            file.write(("{}=$(sbatch --parsable "
                        "--dependency=afterok:$picca_deltas "
                        "{})\n").format(var_xcf, xcf_batch))
            file.write('echo "{0}: "${0} \n'.format(var_xcf))
            file.write(("{}=$(sbatch --parsable "
                        "--dependency=afterok:$picca_deltas "
                        "{})\n").format(var_xdmat, xdmat_batch))
            file.write('echo "{0}: "${0} \n'.format(var_xdmat))
        else:
            file.write(("{}=$(sbatch --parsable "
                        "{})\n").format(var_xcf, xcf_batch))
            file.write('echo "{0}: "${0} \n'.format(var_xcf))
            file.write(
                ("{}=$(sbatch --parsable {})\n").format(var_xdmat, xdmat_batch))
            file.write('echo "{0}: "${0} \n'.format(var_xdmat))
        var_xexp = xexp_batch.replace(".batch",
                                      "").replace(".", "_").replace("-", "_")
        file.write(("{}=$(sbatch --parsable "
                    "--dependency=afterok:${},afterok:${} "
                    "{})\n").format(var_xexp, var_xcf, var_xdmat, xexp_batch))
        file.write('echo "{0}: "${0} \n'.format(var_xexp))

    file.close()
    os.chmod(out_name, stat.S_IRWXU | stat.S_IRWXG)


def main(cmdargs):
    """Writes scripts to send the picca BAO analysis. Once ran, run
    `./submit.sh` in your terminal.

    This script works on nersc::cori not on nersc::edison.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Writes scripts to send the picca BAO analysis. Once ran, '
                     'run `./submit.sh` in your terminal. This script works on '
                     'nersc::cori not on nersc::edison.'))

    parser.add_argument("--out-dir",
                        type=str,
                        default=None,
                        required=True,
                        help="Output directory")

    parser.add_argument("--drq",
                        type=str,
                        default=None,
                        required=True,
                        help="Absolute path to drq file")

    parser.add_argument(
        "--in-dir",
        type=str,
        default=None,
        required=True,
        help=("Absolute path to spectra-NSIDE directory (including "
              "spectra-NSIDE)"))

    parser.add_argument("--email",
                        type=str,
                        default=None,
                        required=False,
                        help="Your email address (optional)")

    parser.add_argument("--to-do",
                        type=str,
                        nargs="*",
                        default=["cf", "xcf"],
                        required=False,
                        help="What to do")

    parser.add_argument("--fid-Om",
                        type=float,
                        default=0.3147,
                        required=False,
                        help="Fiducial Om")

    parser.add_argument("--fid-Or",
                        type=float,
                        default=0.,
                        required=False,
                        help="Fiducial Or")

    parser.add_argument("--fid-Pk",
                        type=str,
                        default="PlanckDR12/PlanckDR12.fits",
                        required=False,
                        help="Fiducial Pk")

    parser.add_argument("--zint",
                        type=str,
                        nargs="*",
                        default=['0:2.35', '2.35:2.65', '2.65:3.05', '3.05:10'],
                        required=False,
                        help="Redshifts intervals")

    parser.add_argument("--mode",
                        type=str,
                        default="desi",
                        required=False,
                        help="Use eboss or desi data")

    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument(
        "--no-deltas",
        action="store_true",
        default=False,
        help="Do not run picca_deltas (e.g. because they were already run)")

    parser.add_argument('--lambda-rest-min',
                        type=float,
                        default=None,
                        required=False,
                        help='Lower limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-max',
                        type=float,
                        default=None,
                        required=False,
                        help='Upper limit on rest frame wavelength [Angstrom]')

    args = parser.parse_args(cmdargs)

    try:
        os.makedirs(args.out_dir + "/deltas")
    except FileExistsError:
        pass

    batch = Batch()
    batch.out_dir = args.out_dir

    time_debug = "00:10:00"
    if "cf" in args.to_do:
        time = "03:30:00"
        if args.debug:
            time = time_debug
        cf_script(batch,
                  time,
                  args.zint,
                  args.out_dir,
                  email=args.email,
                  fid_om=args.fid_om,
                  fid_pk=args.fid_Pk,
                  fid_or=args.fid_Or)

        time = "02:00:00"
        if args.debug:
            time = time_debug
        dmat_script(batch,
                    time,
                    args.zint,
                    args.out_dir,
                    email=args.email,
                    fid_om=args.fid_om,
                    fid_or=args.fid_Or)

    if "xcf" in args.to_do:
        time = "01:30:00"
        if args.debug:
            time = time_debug
        xcf_script(batch,
                   time,
                   args.drq,
                   args.zint,
                   args.out_dir,
                   email=args.email,
                   fid_om=args.fid_om,
                   fid_pk=args.fid_Pk,
                   fid_or=args.fid_Or)

        time = "03:00:00"
        if args.debug:
            time = time_debug
        xdmat_script(batch,
                     time,
                     args.drq,
                     args.zint,
                     args.out_dir,
                     email=args.email,
                     fid_om=args.fid_om,
                     fid_or=args.fid_Or)

    time = "02:00:00"
    if args.debug:
        time = time_debug
    if not args.no_deltas:
        picca_deltas_script(batch,
                            time,
                            args.in_dir,
                            args.out_dir,
                            args.drq,
                            email=args.email,
                            debug=args.debug,
                            mode=args.mode,
                            lambda_rest_frame_min=args.lambda_rest_min,
                            lambda_rest_frame_max=args.lambda_rest_max)

    submit(batch)


if __name__ == "__main__":
    cmdargs=sys.argv[1:]
    main(cmdargs)

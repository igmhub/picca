#!/usr/bin/env python

import os
from os.path import basename, dirname, stat
import argparse

class batch:
    '''
    Generic class to hold the names of the
    .batch slurm submission files.
    '''
    def __init__(self):
        self.outdir = None
        self.picca_deltas = None
        self.cf = []
        self.export = []
        self.dmat = []
        self.xcf = []
        self.xdmat = []
        self.xexport = []

def get_header(time, name, email=None, queue="regular", account="desi"):
    '''
    returns the standard header for running the analyses.
    Input:
        - time (string): requested maximum runtime, format: hh:mm:ss
        - name (string): name of the job
        - email (string, optional): email address to send updates
        - queue (string, optional): name of the submission queue
    Returns:
        a string holding the header
    '''
    header = ""
    header += "#!/bin/bash\n"
    header += "#SBATCH -N 1\n"
    header += "#SBATCH -C haswell\n"
    header += "#SBATCH -q {}\n".format(queue)
    header += "#SBATCH -J {}\n".format(name)
    if email != None:
        header += "#SBATCH --mail-user={}\n".format(email)
        header += "#SBATCH --mail-type=ALL\n"
    header += "#SBATCH -t {}\n".format(time)
    header += "#SBATCH -L project\n"
    header += "#SBATCH -A {}\n".format(account)
    header += "#OpenMP settings:\n"
    header += "export OMP_NUM_THREADS=1\n"

    return header

def picca_deltas(b,time, in_dir, out_dir, drq,
        email=None,debug=False,mode="desi",
        lambda_rest_min=None, lambda_rest_max=None):
    '''
    Writes a .batch file to submit the picca_deltas.py script
    Inputs:
        - b (batch): a batch instance
        - time (string): requested maximum runtime, format: hh:mm:ss
        - in_dir (string): input directory
        - drq (string): drq file
        - email (string, optional): email address to send progress
        - debug (bool, optional): if True, run over 10000 quasars
        - mode (str, optional): desi or eboss (default: desi)
    '''
    assert mode in ["eboss", "desi"]
    if mode=="eboss":
        mode="spplate"
    header = get_header(time, name="picca_deltas", email=email)
    header += "srun -n 1 -c 64 picca_deltas.py " + \
                "--in-dir {} --drq {} ".format(in_dir, drq) + \
                "--out-dir {}/deltas/ --mode {} ".format(out_dir,mode) + \
                "--iter-out-prefix {}/iter --log {}/input.log".format(out_dir,out_dir)
    if not lambda_rest_min is None:
        header += " --lambda-rest-min {}".format(lambda_rest_min)
    if not lambda_rest_max is None:
        header += " --lambda-rest-max {}".format(lambda_rest_max)
    if debug:
        header += " --nspec 10000"

    header += "\n"
    b.picca_deltas = "picca_deltas.batch"
    fout = open(out_dir+"/picca_deltas.batch","w")
    fout.write(header)
    fout.close()

def cf(b,time, zint, outdir, email=None, fidOm = None, fidPk = None, fidOr = None):
    '''
    Writes the .batch files to submit the picca_cf.py
    and the picca_export.py scripts
    and adds them to the b.cf and b.export lists
    Inputs:
        - b (class batch): a batch instance
        - time (string): requested maximum runtime, format: hh:mm:ss
        - zint (list of strings): a list holdint zint-ervals.
            Each interval is of the format "zmin:zmax"
        - outdir (string): the output directory
        - email (string, optional): email address to send progress
    '''
    for zz in zint:
        zmin,zmax = zz.split(":")
        out = "cf_z_{}_{}.fits".format(zmin,zmax)
        exp_batch = export("00:10:00",
                outdir+"/cf_z_{}_{}.fits".format(zmin,zmax),
                outdir+"/dmat_z_{}_{}.fits".format(zmin,zmax),
                outdir+"/cf_z_{}_{}-exp.fits".format(zmin,zmax),
                fidPk=fidPk)
        header = get_header(time, name=out, email=email)
        srun = header + "srun -n 1 -c 64 picca_cf.py --in-dir {}/deltas/ ".format(outdir) +\
                "--z-cut-min {} --z-cut-max {} ".format(zmin,zmax) +\
                "--out {}/{} --nproc 32 --fid-Om {} --fid-Or {} \n".format(outdir,out,fidOm,fidOr)

        fbatch = outdir+"/"+out.replace(".fits",".batch")
        b.cf.append(basename(fbatch))
        b.export.append(basename(exp_batch))

        fout = open(fbatch,"w")
        fout.write(srun)
        fout.close()

def dmat(b,time, zint, outdir, email=None, rej=0.99, fidOm=None, fidOr = None):
    '''
    Writes the .batch files to submit the picca_dmat.py script
    and adds them to the b.dmat script
    Inputs:
        - b (class batch): a batch instance
        - time (string): requested maximum runtime, format: hh:mm:ss
        - zint (list of strings): a list holdint zint-ervals.
            Each interval is of the format "zmin:zmax"
        - outdir (string): the output directory
        - email (string, optional): email address to send progress
    '''
    for zz in zint:
        zmin,zmax = zz.split(":")
        out = "dmat_z_{}_{}.fits".format(zmin,zmax)
        header = get_header(time, name=out, email=email)
        srun = header + "srun -n 1 -c 64 picca_dmat.py --in-dir {}/deltas/ ".format(outdir) +\
                "--z-cut-min {} --z-cut-max {} ".format(zmin,zmax) +\
                "--out {}/{} --rej {} --nproc 32 --fid-Om {} --fid-Or {}\n".format(outdir,out,rej,fidOm,fidOr)
        fbatch = outdir+"/"+out.replace(".fits",".batch")
        b.dmat.append(basename(fbatch))

        fout = open(fbatch,"w")
        fout.write(srun)
        fout.close()

def xcf(b,time, drq, zint, outdir, email=None,fidOm=None, fidPk=None, fidOr = None):
    '''
    Writes the .batch files to submit the picca_xcf.py script
    and the picca_export.py scripts
    and adds them to the b.xcf and b.xexport lists
    Inputs:
        - b (class batch): a batch instance
        - time (string): requested maximum runtime, format: hh:mm:ss
        - zint (list of strings): a list holdint zint-ervals.
            Each interval is of the format "zmin:zmax"
        - outdir (string): the output directory
        - email (string, optional): email address to send progress
    '''
    for zz in zint:
        zmin,zmax = zz.split(":")
        out = "xcf_z_{}_{}.fits".format(zmin,zmax)
        header = get_header(time, name=out, email=email)
        exp_batch = export("00:10:00",
                outdir+"/xcf_z_{}_{}.fits".format(zmin,zmax),
                outdir+"/xdmat_z_{}_{}.fits".format(zmin,zmax),
                outdir+"/xcf_z_{}_{}-exp.fits".format(zmin,zmax),
                fidPk=fidPk)
        srun = header + "srun -n 1 -c 64 picca_xcf.py " +\
            "--drq {} --in-dir {}/deltas/ ".format(drq,outdir) +\
             "--z-evol-obj 1.44 --z-cut-min {} --z-cut-max {} ".format(zmin, zmax) +\
             "--out {}/{} --nproc 32 --fid-Om {} --fid-Or {}\n".format(outdir,out,fidOm,fidOr)
        fbatch = outdir+"/"+out.replace(".fits",".batch")
        b.xcf.append(basename(fbatch))
        b.xexport.append(basename(exp_batch))

        fout = open(fbatch,"w")
        fout.write(srun)
        fout.close()

def xdmat(b,time, drq, zint, outdir, email=None, rej=0.95,fidOm=None, fidOr = None):
    '''
    Writes the .batch files to submit the picca_xdmat.py script
    and adds if to the b.xdmat list
    Inputs:
        - b (class batch): a batch instance
        - time (string): requested maximum runtime, format: hh:mm:ss
        - zint (list of strings): a list holdint zint-ervals.
            Each interval is of the format "zmin:zmax"
        - outdir (string): the output directory
        - email (string, optional): email address to send progress
    '''
    for zz in zint:
        zmin, zmax = zz.split(":")
        out = "xdmat_z_{}_{}.fits".format(zmin,zmax)
        header = get_header(time, name=out, email=email)
        srun = header + "srun -n 1 -c 64 picca_xdmat.py " +\
            "--drq {} --in-dir {}/deltas/ ".format(drq,outdir) +\
            "--z-evol-obj 1.44 --z-cut-min {} --z-cut-max {} ".format(zmin, zmax) +\
            "--out {}/{} --rej {} --nproc 32 --fid-Om {} --fid-Or {}\n".format(outdir,out,rej,fidOm,fidOr)
        fbatch = outdir+"/"+out.replace(".fits",".batch")
        b.xdmat.append(basename(fbatch))

        fout = open(fbatch,"w")
        fout.write(srun)
        fout.close()

def export(time, cf_file, dmat_file, out, fidPk):
    '''
    Writes the .batch file to submit the picca_export.py and picca_fitter2.py
    Input:
        - time (string): requested maximum runtime, format: hh:mm:ss
        - cf_file (string): path to the cf_file
        - dmat_file (string): path to the dmat_file
        - out (string): output of the picca_export.py script
    '''
    header = get_header(time, name=basename(out), queue="regular")
    srun = header + "srun -n 1 -c 64 picca_export.py "+\
            "--data {} --dmat {} ".format(cf_file,dmat_file)+\
            "--out {}\n".format(out)
    chi2_ini = do_ini(dirname(out), basename(out),fidPk)
    srun += "srun -n 1 -c 64 picca_fitter2.py {}\n".format(chi2_ini)
    fbatch = out.replace(".fits",".batch")
    fout = open(fbatch,"w")
    fout.write(srun)
    fout.close()

    return fbatch

def do_ini(outdir, cf_file,fidPk):
    '''
    Writes the .ini files to control the fits
    Input:
        - outdir (string): output directory of the fits
        - cf_file (string): path to the input file for the fitter
    Returns:
        The (basename) of the chi2.ini file
    '''
    fout = open(outdir+"/"+cf_file.replace(".fits",".ini"),"w")
    fout.write("[data]\n")
    fout.write("name = {}\n".format(basename(cf_file).replace(".fits","")))
    fout.write("tracer1 = LYA\n")
    fout.write("tracer1-type = continuous\n")
    if "xcf" in cf_file:
        fout.write("tracer2 = QSO\n")
        fout.write("tracer2-type = discrete\n")
    else:
        fout.write("tracer2 = LYA\n")
        fout.write("tracer2-type = continuous\n")
    fout.write("filename = {}\n".format(outdir+'/'+cf_file))
    fout.write("ell-max = 6\n")

    fout.write("[cuts]\n")
    fout.write("rp-min = -200\n")
    fout.write("rp-max = 200\n")

    fout.write("rt-min = 0\n")
    fout.write("rt-max = 200\n")

    fout.write("r-min = 10\n")
    fout.write("r-max = 180\n")

    fout.write("mu-min = -1\n")
    fout.write("mu-max = 1\n")

    fout.write("[model]\n")
    fout.write("model-pk = pk_kaiser\n")
    fout.write("z evol LYA = bias_vs_z_std\n")
    fout.write("growth function = growth_factor_de\n")
    if "xcf" in cf_file:
        fout.write("model-xi = xi_drp\n")
        fout.write("z evol QSO = bias_vs_z_std\n")
        fout.write("velocity dispersion = pk_velo_lorentz\n")
    else:
        fout.write("model-xi = xi\n")


    fout.write("[parameters]\n")

    fout.write("ap = 1. 0.1 0.5 1.5 free\n")
    fout.write("at = 1. 0.1 0.5 1.5 free\n")
    fout.write("bias_eta_LYA = -0.17 0.017 None None free\n")
    fout.write("beta_LYA = 1. 0.1 None None free\n")
    fout.write("alpha_LYA = 2.9 0.1 None None fixed\n")
    if "xcf" in cf_file:
        fout.write("bias_eta_QSO = 1. 0.1 None None fixed\n")
        fout.write("beta_QSO = 0.3 0.1 None None fixed\n")
        fout.write("alpha_QSO = 1.44 0.1 None None fixed\n")

    fout.write("growth_rate = 0.962524 0.1 None None fixed\n")

    fout.write("sigmaNL_par = 6.36984 0.1 None None fixed\n")
    fout.write("sigmaNL_per = 3.24 0.1 None None fixed\n")

    fout.write("par binsize {} = 4. 0.4 None None free\n".format(cf_file.replace(".fits","")))
    fout.write("per binsize {} = 4. 0.4 None None free\n".format(cf_file.replace(".fits","")))

    fout.write("bao_amp = 1. 0.1 None None fixed\n")
    if "xcf" in cf_file:
        fout.write("drp_QSO = 0. 0.1 None None free\n")
        fout.write("sigma_velo_lorentz_QSO = 5. 0.1 None None free\n")
    fout.close()

    chi2_ini = outdir+"/chi2_{}".format(cf_file.replace(".fits",".ini"))
    fout = open(chi2_ini,"w")
    fout.write("[data sets]\n")
    fout.write("zeff = 2.310\n")
    fout.write("ini files = {}\n".format(outdir+"/"+cf_file.replace(".fits",".ini")))

    fout.write("[fiducial]\n")
    fout.write("filename = {}\n".format(fidPk))

    fout.write("[verbosity]\n")
    fout.write("level = 0\n")

    fout.write("[output]\n")
    fout.write("filename = {}\n".format(outdir+"/"+cf_file.replace(".fits",".h5")))

    fout.write("[cosmo-fit type]\n")
    fout.write("cosmo fit func = ap_at\n")
    fout.close()

    return chi2_ini

def submit(b):
    out_name = b.outdir+"/submit.sh"
    fout = open(out_name,"w")
    fout.write("#!/bin/bash\n")
    if b.picca_deltas is not None:
        fout.write("picca_deltas=$(sbatch --parsable {})\n".format(b.picca_deltas))
        fout.write('echo "picca_deltas: "$picca_deltas\n')
    for cf_batch, dmat_batch,exp_batch in zip(b.cf, b.dmat, b.export):
        var_cf = cf_batch.replace(".batch","").replace(".","_")
        var_dmat = dmat_batch.replace(".batch","").replace(".","_")
        if b.picca_deltas is not None:
            fout.write("{}=$(sbatch --parsable --dependency=afterok:$picca_deltas {})\n".format(var_cf,cf_batch))
            fout.write('echo "{0}: "${0} \n'.format(var_cf))
            fout.write("{}=$(sbatch --parsable --dependency=afterok:$picca_deltas {})\n".format(var_dmat,dmat_batch))
            fout.write('echo "{0}: "${0} \n'.format(var_dmat))
        else:
            fout.write("{}=$(sbatch --parsable {})\n".format(var_cf,cf_batch))
            fout.write('echo "{0}: "${0} \n'.format(var_cf))
            fout.write("{}=$(sbatch --parsable {})\n".format(var_dmat,dmat_batch))
            fout.write('echo "{0}: "${0} \n'.format(var_dmat))
        var_exp = exp_batch.replace(".batch","").replace(".","_").replace("-","_")
        fout.write("{}=$(sbatch --parsable --dependency=afterok:${},afterok:${} {})\n".format(var_exp,var_cf,var_dmat,exp_batch))
        fout.write('echo "{0}: "${0} \n'.format(var_exp))

    for xcf_batch, xdmat_batch,xexp_batch in zip(b.xcf, b.xdmat, b.xexport):
        var_xcf = xcf_batch.replace(".batch","").replace(".","_")
        var_xdmat = xdmat_batch.replace(".batch","").replace(".","_")
        if b.picca_deltas is not None:
            fout.write("{}=$(sbatch --parsable --dependency=afterok:$picca_deltas {})\n".format(var_xcf,xcf_batch))
            fout.write('echo "{0}: "${0} \n'.format(var_xcf))
            fout.write("{}=$(sbatch --parsable --dependency=afterok:$picca_deltas {})\n".format(var_xdmat,xdmat_batch))
            fout.write('echo "{0}: "${0} \n'.format(var_xdmat))
        else:
            fout.write("{}=$(sbatch --parsable {})\n".format(var_xcf,xcf_batch))
            fout.write('echo "{0}: "${0} \n'.format(var_xcf))
            fout.write("{}=$(sbatch --parsable {})\n".format(var_xdmat,xdmat_batch))
            fout.write('echo "{0}: "${0} \n'.format(var_xdmat))
        var_xexp = xexp_batch.replace(".batch","").replace(".","_").replace("-","_")
        fout.write("{}=$(sbatch --parsable --dependency=afterok:${},afterok:${} {})\n".format(var_xexp,var_xcf,var_xdmat,xexp_batch))
        fout.write('echo "{0}: "${0} \n'.format(var_xexp))

    fout.close()
    os.chmod(out_name,stat.S_IRWXU | stat.S_IRWXG)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Writes scripts to send the picca BAO analysis. Once ran, run `./submit.sh` '+\
                'in your terminal. This script works on nersc::cori not on nersc::edison.')

parser.add_argument("--out-dir", type=str, default=None, required=True,
        help="Output directory")

parser.add_argument("--drq", type=str,default=None, required=True,
        help="Absolute path to drq file")

parser.add_argument("--in-dir", type=str, default=None, required=True,
        help="Absolute path to spectra-NSIDE directory "+\
                "(including spectra-NSIDE)")

parser.add_argument("--email", type=str, default=None, required=False,
        help="Your email address (optional)")

parser.add_argument("--to-do", type=str, nargs="*",
        default=["cf","xcf"],
        required=False, help="What to do")

parser.add_argument("--fid-Om", type=float,
        default=0.3147,
        required=False, help="Fiducial Om")

parser.add_argument("--fid-Or", type=float,
        default=0.,
        required=False, help="Fiducial Or")

parser.add_argument("--fid-Pk", type=str,
        default="PlanckDR12/PlanckDR12.fits",
        required=False, help="Fiducial Pk")

parser.add_argument("--zint", type=str, nargs="*",
        default=['0:2.35','2.35:2.65','2.65:3.05','3.05:10'],
        required=False, help="Redshifts intervals")

parser.add_argument("--mode", type=str,
        default="desi",
        required=False, help="Use eboss or desi data")

parser.add_argument("--debug",
        action="store_true", default=False)

parser.add_argument("--no-deltas",
        action="store_true", default=False,
        help="Do not run picca_deltas (e.g. because they were already run)")

parser.add_argument('--lambda-rest-min',type=float,default=None,required=False,
        help='Lower limit on rest frame wavelength [Angstrom]')

parser.add_argument('--lambda-rest-max',type=float,default=None,required=False,
        help='Upper limit on rest frame wavelength [Angstrom]')

args = parser.parse_args()

try:
    os.makedirs(args.out_dir+"/deltas")
except FileExistsError:
    pass

b = batch()
b.outdir = args.out_dir

time_debug = "00:10:00"
if "cf" in args.to_do:
    time = "03:30:00"
    if args.debug:
        time = time_debug
    cf(b,time, args.zint, args.out_dir,
            email=args.email,fidOm=args.fid_Om,fidPk=args.fid_Pk, fidOr=args.fid_Or)

    time = "02:00:00"
    if args.debug:
        time = time_debug
    dmat(b,time, args.zint, args.out_dir,
            email=args.email,fidOm=args.fid_Om, fidOr=args.fid_Or)

if "xcf" in args.to_do:
    time = "01:30:00"
    if args.debug:
        time = time_debug
    xcf(b,time, args.drq, args.zint, args.out_dir,
            email=args.email,fidOm=args.fid_Om, fidPk=args.fid_Pk, fidOr=args.fid_Or)

    time = "03:00:00"
    if args.debug:
        time = time_debug
    xdmat(b,time, args.drq, args.zint, args.out_dir,
            email=args.email, fidOm=args.fid_Om, fidOr=args.fid_Or)

time = "02:00:00"
if args.debug:
    time = time_debug
if not args.no_deltas:
    picca_deltas(b,time,args.in_dir, args.out_dir,args.drq,
            email=args.email, debug=args.debug, mode=args.mode,
        lambda_rest_min=args.lambda_rest_min, lambda_rest_max=args.lambda_rest_max)

submit(b)

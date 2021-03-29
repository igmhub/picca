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
        self.stack = None
        self.xstack = None

def get_header(time, outdir, name, email=None, queue="regular", account="desi"):
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
    header += "#SBATCH --output={}/logs/{}.logs\n".format(outdir, name.replace('.fits',''))
    if email != None:
        header += "#SBATCH --mail-user={}\n".format(email)
        header += "#SBATCH --mail-type=ALL\n"
    header += "#SBATCH -t {}\n".format(time)
    header += "#SBATCH -L project\n"
    header += "#SBATCH -A {}\n".format(account)
    header += "#OpenMP settings:\n"
    header += "export OMP_NUM_THREADS=1\n"

    return header

def picca_deltas(b,time, in_dir, out_dir, drq, email=None,debug=False,mode="desi", lambda_rest_min=None, lambda_rest_max=None, mask_dla_cat=None, use_constant_weight=None):
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
    assert mode in ["eboss", "desi", "spplate"]
    
    if 'raw' in out_dir:
        header = get_header(time, b.outdir, name="picca_deltas",  email=email)
        header += "/usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' srun -n 1 -c 64 delta_from_transmission.py " + \
                    "--in-dir {} ".format(in_dir) + \
                    "--zcat {} ".format(drq)+ \
                    "--out-dir {}/deltas/ ".format(out_dir)

    elif 'true' in out_dir:
        header = get_header(time, b.outdir, name="deltas_from_true_cont",  email=email)
        header += "/usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' srun -n 1 -c 64 delta_from_true_cont.py " + \
                    "--in-dir_trans {} ".format(in_dir.replace("eboss-raw/",'').replace("spectra-16/",'')) + \
                    "--in-dir_spec {} --zcat {}".format(in_dir, drq) + \
                    "--out-dir {}/deltas/ ".format(out_dir)
    else:
        header = get_header(time, b.outdir, name="picca_deltas",  email=email)
        header += "/usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' srun -n 1 -c 64 picca_deltas.py " + \
                    "--in-dir {} --drq {} ".format(in_dir, drq) + \
                    "--out-dir {}/deltas/ --mode {} ".format(out_dir,mode) + \
                    "--iter-out-prefix {}/iter --log {}/input.log".format(out_dir,out_dir)
        
    if not lambda_rest_min is None:
        header += " --lambda-rest-min {}".format(lambda_rest_min)
    if not lambda_rest_max is None:
        header += " --lambda-rest-max {}".format(lambda_rest_max)
    if not mask_dla_cat is None:
        header += " --dla-vac {}".format(mask_dla_cat)
    
    if debug:
        header += " --nspec 10000"
        
    if use_constant_weight:
        header += " --use-constant-weight"
        
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
        if '0.3' in outdir:
            time_exp = "00:50:00"
        else:
            time_exp = "00:10:00"
        
        if args.dmat_file is None and 'raw' not in outdir and 'true' not in outdir:
            exp_batch = export(time_exp,
                    outdir+"/cf_z_{}_{}.fits".format(zmin,zmax),
                    outdir+"/dmat_z_{}_{}.fits".format(zmin,zmax),
                    outdir+"/cf_z_{}_{}-exp.fits".format(zmin,zmax),
                    fidPk=fidPk)
        elif 'raw' in outdir or 'true' in outdir:
            exp_batch = export(time_exp,
                    outdir+"/cf_z_{}_{}.fits".format(zmin,zmax),
                    None,
                    outdir+"/cf_z_{}_{}-exp.fits".format(zmin,zmax),
                    fidPk=fidPk)
            
        else:
            exp_batch = export(time_exp,
                    outdir+"/cf_z_{}_{}.fits".format(zmin,zmax),
                    args.dmat_file+"/dmat_z_{}_{}.fits".format(zmin,zmax),
                    outdir+"/cf_z_{}_{}-exp.fits".format(zmin,zmax),
                    fidPk=fidPk)
        header = get_header(time, b.outdir, name=out, email=email)
        srun = header + "/usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' srun -n 1 -c 64 picca_cf.py --in-dir {}/deltas/ ".format(outdir) +\
                "--z-cut-min {} --z-cut-max {} ".format(zmin,zmax) +\
                "--out {}/{} --nproc 32 --fid-Om {} --fid-Or {}".format(outdir,out,fidOm,fidOr)
        if 'raw' in outdir or 'true' in outdir:
            srun += ' --no-project \n'

        fbatch = outdir+"/"+out.replace(".fits",".batch")
        b.cf.append(basename(fbatch))
        b.export.append(basename(exp_batch))

        fout = open(fbatch,"w")
        fout.write(srun)
        fout.close()

    if len(zint) >1:
        stack_batch = stack(b,"00:20:00", zint, outdir, '', fidPk)
        b.stack = "cf_z_0_10-exp.batch"

def dmat(b,time, zint, outdir, email=None, rej=0.95, fidOm=None, fidOr = None):
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
        header = get_header(time, b.outdir, name=out, email=email)
        srun = header + "/usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' srun -n 1 -c 64 picca_dmat.py --in-dir {}/deltas/ ".format(outdir) +\
                "--z-cut-min {} --z-cut-max {} ".format(zmin,zmax) +\
                "--out {}/{} --rej {} --nproc 32 --fid-Om {} --fid-Or {}\n".format(outdir,out,rej,fidOm,fidOr)
        fbatch = outdir+"/"+out.replace(".fits",".batch")
        b.dmat.append(basename(fbatch))
        
        if args.dmat_file is None and 'raw' not in outdir and 'true' not in outdir:
            fout = open(fbatch,"w")
            fout.write(srun)
            fout.close()

def xcf(b,time, drq, zint, outdir, email=None,fidOm=None, fidPk=None, fidOr = None, shuffle_seed = None):
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
        if '0.3' in outdir:
            time_exp = "00:50:00"
        else:
            time_exp = "00:10:00"
        
        if args.dmat_file is None and 'raw' not in outdir and 'true' not in outdir:
            exp_batch = export(time_exp,
                outdir+"/xcf_z_{}_{}.fits".format(zmin,zmax),
                outdir+"/xdmat_z_{}_{}.fits".format(zmin,zmax),
                outdir+"/xcf_z_{}_{}-exp.fits".format(zmin,zmax),
                shuffle = outdir+"/xcf_z_{}_{}_shuffle.fits".format(zmin,zmax),
                fidPk=fidPk)
        
        elif 'raw' in outdir or 'true' in outdir:
            exp_batch = export(time_exp,
                outdir+"/xcf_z_{}_{}.fits".format(zmin,zmax),
                None,
                outdir+"/xcf_z_{}_{}-exp.fits".format(zmin,zmax),
                shuffle = outdir+"/xcf_z_{}_{}_shuffle.fits".format(zmin,zmax),
                fidPk=fidPk)
            
        else:
            exp_batch = export(time_exp,
                    outdir+"/xcf_z_{}_{}.fits".format(zmin,zmax),
                    args.dmat_file+"/xdmat_z_{}_{}.fits".format(zmin,zmax),
                    outdir+"/xcf_z_{}_{}-exp.fits".format(zmin,zmax),
                    shuffle = outdir+"/xcf_z_{}_{}_shuffle.fits".format(zmin,zmax),
                    fidPk=fidPk)
        header = get_header(time, b.outdir, name=out, email=email)

        srun = header + "/usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' srun -n 1 -c 64 picca_xcf.py " +\
                "--drq {} --in-dir {}/deltas/ ".format(drq,outdir) +\
                 "--z-evol-obj 1.44 --z-cut-min {} --z-cut-max {} ".format(zmin, zmax) +\
                 "--out {}/{} --nproc 32 --fid-Om {} --fid-Or {}".format(outdir,out,fidOm,fidOr)
        if 'raw' in outdir or 'true' in outdir: 
            srun += ' --no-project\n'
        
        #run additional correlation with shuffled pairs 
        srun += "\n /usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' srun -n 1 -c 64 picca_xcf.py " +\
                "--drq {} --in-dir {}/deltas/ ".format(drq,outdir) +\
                 "--z-evol-obj 1.44 --z-cut-min {} --z-cut-max {} ".format(zmin, zmax) +\
                 "--out {}/{} --nproc 32 --fid-Om {} --fid-Or {} --shuffle-distrib-obj-seed {}".format(outdir, out.replace('.fits', '_shuffle.fits'), fidOm, fidOr, shuffle_seed)
        if 'raw' in outdir or 'true' in outdir: 
            srun += ' --no-project\n'


        fbatch = outdir+"/"+out.replace(".fits",".batch")
        b.xcf.append(basename(fbatch))
        b.xexport.append(basename(exp_batch))

        fout = open(fbatch,"w")
        fout.write(srun)
        fout.close()
        
    if len(zint) >1:
        stack_batch = stack(b,"00:20:00", zint, outdir, 'x', fidPk)
        b.xstack = "xcf_z_0_10-exp.batch"

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
        header = get_header(time, b.outdir, name=out, email=email)
        srun = header + "/usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' srun -n 1 -c 64 picca_xdmat.py " +\
            "--drq {} --in-dir {}/deltas/ ".format(drq,outdir) +\
            "--z-evol-obj 1.44 --z-cut-min {} --z-cut-max {} ".format(zmin, zmax) +\
            "--out {}/{} --rej {} --nproc 32 --fid-Om {} --fid-Or {}\n".format(outdir,out,rej,fidOm,fidOr)
        fbatch = outdir+"/"+out.replace(".fits",".batch")
        b.xdmat.append(basename(fbatch))
        
        if args.dmat_file is None and 'raw' not in outdir and 'true' not in outdir:
            fout = open(fbatch,"w")
            fout.write(srun)
            fout.close()
        
def stack(b,time, zint, outdir, cor, fidPk, email=None):
    '''
    Writes the .batch files to submit calculate the stack
    and adds them to the b.cf and b.export lists
    Inputs:
        - b (class batch): a batch instance
        - time (string): requested maximum runtime, format: hh:mm:ss
        - zint (list of strings): a list holdint zint-ervals.
            Each interval is of the format "zmin:zmax"
        - outdir (string): the output directory
        - email (string, optional): email address to send progress
    ''' 
    name = '/{}cf_z_0_10-exp.fits'.format(cor)
    out = outdir+name
    header = get_header(time, b.outdir, name=name, email=email)
    srun = header + "/usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' srun -n 1 -c 64 /global/homes/j/jstermer/programs/picca/tutorials/picca_export_stacked_correlation.py --out {}".format(out)
    srun += " --data "
    for zz in zint:
        zmin,zmax = zz.split(":")
        srun += outdir+'/{}cf_z_{}_{}.fits '.format(cor,zmin,zmax)
    
    if 'raw' not in outdir and 'true' not in outdir:
        srun += ' --dmat '
        for zz in zint:
            zmin,zmax = zz.split(":")
            if args.dmat_file is None:
                srun += outdir+'/{}dmat_z_{}_{}.fits '.format(cor,zmin,zmax)

            else:
                srun += args.dmat_file+'/{}dmat_z_{}_{}.fits '.format(cor,zmin,zmax)
    srun+='\n'

    chi2_ini = do_ini(dirname(out), basename(out),fidPk)
    
    srun += "srun -n 1 -c 64 /usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' picca_zeff.py --data {} --chi2 {}\n".format(out, chi2_ini)
    srun += "srun -n 1 -c 64 /usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' picca_fitter2.py {}\n".format(chi2_ini)
    
    fbatch = "/"+out.replace(".fits",".batch")
    fout = open(fbatch,"w")
    fout.write(srun)
    fout.close()

def export(time, cf_file, dmat_file, out, fidPk, shuffle = None):
    '''
    Writes the .batch file to submit the picca_export.py and picca_fitter2.py
    Input:
        - time (string): requested maximum runtime, format: hh:mm:ss
        - cf_file (string): path to the cf_file
        - dmat_file (string): path to the dmat_file
        - out (string): output of the picca_export.py script
    '''
    header = get_header(time, b.outdir, name=basename(out), queue="regular")
    if dmat_file is None:
        srun = header + "srun -n 1 -c 64 picca_export.py"+\
                " --data {}".format(cf_file)+\
                " --out {}".format(out)
    else:
        srun = header + "srun -n 1 -c 64 picca_export.py"+\
                " --data {} --dmat {}".format(cf_file,dmat_file)+\
                " --out {}".format(out)
    if not shuffle is None:
        srun += " --remove-shuffled-correlation {}\n".format(shuffle)
    else:
        srun +='\n'
        
    
    chi2_ini = do_ini(dirname(out), basename(out),fidPk)
    
    srun += "srun -n 1 -c 64 /usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' picca_zeff.py --data {} --chi2 {}\n".format(out, chi2_ini)
    
    srun += "srun -n 1 -c 64 /usr/bin/time -f '%eReal %Uuser %Ssystem %PCPU %M' picca_fitter2.py {}\n".format(chi2_ini)
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
    fout.write("\n")
    fout.write("[cuts]\n")
    fout.write("rp-min = -200.\n")
    fout.write("rp-max = 200.\n")
    fout.write("\n")
    fout.write("rt-min = 0.\n")
    fout.write("rt-max = 200.\n")
    fout.write("\n")
    fout.write("r-min = 20.\n")
    fout.write("r-max = 180.\n")
    fout.write("\n")
    fout.write("mu-min = -1.\n")
    fout.write("mu-max = 1.\n")

    fout.write("\n")
    fout.write("[model]\n")
    
    if "0.2" in outdir or "0.3" in outdir:
        if 'xcf' in cf_file:
            fout.write("model-pk = pk_hcd_Rogers2018_cross\n")
        else:
            fout.write("model-pk = pk_hcd_Rogers2018\n")
    else:
        fout.write("model-pk = pk_kaiser\n")
        
    fout.write("z evol LYA = bias_vs_z_std\n")
    fout.write("growth function = growth_factor_de\n")
    fout.write("pk-gauss-smoothing = pk_gauss_smoothing\n")
    if "xcf" in cf_file:
        fout.write("model-xi = xi_drp\n")
        fout.write("z evol QSO = bias_vs_z_std\n")
        fout.write("velocity dispersion = pk_velo_lorentz\n")
    else:
        fout.write("model-xi = xi\n")

    fout.write("\n")
    fout.write("[parameters]\n")

    fout.write("ap = 1. 0.1 0.5 1.5 free\n")
    fout.write("at = 1. 0.1 0.5 1.5 free\n")
    fout.write("bao_amp = 1. 0.1 None None fixed\n")
    fout.write("\n")
    
    if 'london' in outdir:
        fout.write("growth_rate = 0.970386193694752 0. None None fixed\n")
        fout.write("sigmaNL_par = 6.36984 0.1 None None fixed\n")
        fout.write("sigmaNL_per = 3.24 0.1 None None fixed\n")
    else:
        fout.write("growth_rate = 0.970386193694752 0. None None fixed\n")
        fout.write("sigmaNL_par = 0 0 None None fixed\n")
        fout.write("sigmaNL_per = 0 0 None None fixed\n")
    
    if "xcf" in cf_file:
        fout.write("\n")
        fout.write("drp_QSO = 0. 0.1 None None free\n")
        if 'london' in outdir:
            fout.write("sigma_velo_lorentz_QSO = 5. 0.1 None None free\n")
        else:
            fout.write("sigma_velo_lorentz_QSO = 0 0.1 None None fixed\n")
    
    fout.write("\n")
    fout.write("par binsize {} = 4. 0.4 None None fixed\n".format(cf_file.replace(".fits","")))
    fout.write("per binsize {} = 4. 0.4 None None fixed\n".format(cf_file.replace(".fits","")))
    fout.write("\n")
    fout.write("bias_eta_LYA = -0.17 0.017 None None free\n")
    fout.write("beta_LYA = 1.8 0.2 None None free\n")
    fout.write("alpha_LYA = 2.9 0.1 None None fixed\n")
    
    fout.write("par_sigma_smooth = 2.19 0 0 None fixed\n")
    fout.write("per_sigma_smooth = 2.19 0 0 None fixed\n")
    
    if "xcf" in cf_file:
        fout.write("\n")
        fout.write("bias_eta_QSO = 1. 0.1 None None fixed\n")
        fout.write("beta_QSO = 0.3 0.1 None None fixed\n")
        fout.write("alpha_QSO = 1.7 0.1 None None fixed\n")
    
    if "0.2" in outdir or "0.3" in outdir:
        fout.write("\n")
        fout.write("bias_hcd = -1.68E-2 0.1 None None free\n")
        fout.write("beta_hcd = 0.67 0.1 None None free\n")
        fout.write("L0_hcd = 10.0 1 None None fixed\n")
    
    if "0.1" in outdir or "0.3" in outdir:
        fout.write("\n")
        fout.write("bias_eta_SiII(1260) = -0.60E-3 0.01 None None free\n")
        fout.write("beta_SiII(1260) = 0.5 0. None None fixed\n")
        fout.write("alpha_SiII(1260) = 1.0 0. None None fixed\n")
        fout.write("\n")
        fout.write("bias_eta_SiIII(1207) = -1.74E-3 0.01 None None free\n")
        fout.write("beta_SiIII(1207) = 0.5 0. None None fixed\n")
        fout.write("alpha_SiIII(1207) = 1.0 0. None None fixed\n")
        fout.write("\n")
        fout.write("bias_eta_SiII(1193) = -1.08E-3 0.01 None None free\n")
        fout.write("beta_SiII(1193) = 0.5 0. None None fixed\n")
        fout.write("alpha_SiII(1193) = 1.0 0. None None fixed\n")
        fout.write("\n")
        fout.write("bias_eta_SiII(1190) = -0.95E-3 0.01 None None free\n")
        fout.write("beta_SiII(1190) = 0.5 0. None None fixed\n")
        fout.write("alpha_SiII(1190) = 1.0 0. None None fixed\n")
        fout.write("\n")
        fout.write("bias_eta_CIV(eff) = -0.00513 0.001 None 0. free\n")
        fout.write("beta_CIV(eff) = 0.27 0.01 None 1. fixed\n")
        fout.write("alpha_CIV(eff) = 1. 0.01 None None fixed\n")
        
        fout.write("\n")
        fout.write("[metals]\n")
        
        fout.write("filename = {}\n".format(outdir+'/'+cf_file).replace('cf_z', 'metal_dmat_z').replace('-exp',''))
        fout.write("model-pk-met = pk_kaiser\n")
        fout.write("model-xi-met = xi\n")
        fout.write("z evol = bias_vs_z_std\n")
        fout.write("in tracer1 = CIV(eff) SiII(1260) SiIII(1207) SiII(1193) SiII(1190)\n")
        fout.write("in tracer2 = CIV(eff) SiII(1260) SiIII(1207) SiII(1193) SiII(1190)\n")
    
    if "0.1" in outdir or "0.2" in outdir or "0.3" in outdir:
        fout.write("\n")
        fout.write("[priors]\n")
        if "0.2" in outdir or "0.3" in outdir:
            fout.write("beta_hcd = gaussian 0.5 0.09\n")
        if "0.1" in outdir or "0.3" in outdir:
            fout.write("bias_eta_CIV(eff) = gaussian -0.005 0.0026\n")
    
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
        fout.write("picca_deltas=$(sbatch --parsable {})\n".format(b.outdir+'/'+b.picca_deltas))
        fout.write('echo "picca_deltas: "$picca_deltas\n')
    
    for cor in [[b.cf, b.dmat, b.export],[b.xcf, b.xdmat, b.xexport]]:
        for cf_batch, dmat_batch,exp_batch in zip(cor[0], cor[1], cor[2]):
            var_cf = cf_batch.replace(".batch","").replace(".","_")
            var_dmat = dmat_batch.replace(".batch","").replace(".","_")
            if b.picca_deltas is not None:
                fout.write("{}=$(sbatch --parsable --dependency=afterok:$picca_deltas {})\n".format(var_cf, b.outdir+'/'+cf_batch))
                fout.write('echo "{0}: "${0} \n'.format(var_cf))
                if args.dmat_file is None and 'raw' not in b.outdir and 'true' not in b.outdir:
                    fout.write("{}=$(sbatch --parsable --dependency=afterok:$picca_deltas {})\n".format(var_dmat, b.outdir+'/'+dmat_batch))
                    fout.write('echo "{0}: "${0} \n'.format(var_dmat))
            else:
                fout.write("{}=$(sbatch --parsable {})\n".format(var_cf, b.outdir+'/'+cf_batch))
                fout.write('echo "{0}: "${0} \n'.format(var_cf))
                if args.dmat_file is None and 'raw' not in b.outdir and 'true' not in b.outdir:
                    fout.write("{}=$(sbatch --parsable {})\n".format(var_dmat, b.outdir+'/'+dmat_batch))
                    fout.write('echo "{0}: "${0} \n'.format(var_dmat))
            
            var_exp = exp_batch.replace(".batch","").replace(".","_").replace("-","_")
            
            if args.dmat_file is None and 'raw' not in b.outdir and 'true' not in b.outdir:
                fout.write("{}=$(sbatch --parsable --dependency=afterok:${},afterok:${} {})\n".format(var_exp,var_cf,var_dmat, b.outdir+'/'+exp_batch))
            else:
                fout.write("{}=$(sbatch --parsable --dependency=afterok:${} {})\n".format(var_exp,var_cf, b.outdir+'/'+exp_batch))
            fout.write('echo "{0}: "${0} \n'.format(var_exp))
    
    _stacks = []
    if b.stack is not None:
        _stacks.append(b.stack)
    if b.stack is not None:
        _stacks.append(b.xstack)
    if len(_stacks)  >=1:
        for stack in _stacks:
            print(stack)
            var_stack = stack.replace(".batch","").replace(".","_").replace("-","_")
            text ='{}=$(sbatch --parsable --dependency='.format(var_stack)
            num = 1  ### pb with commas for dependencies need to find better way
            
            if 'xcf' in stack :
                cor = [b.xcf,b.xdmat]
            else:
                cor = [b.cf,b.dmat]
                
            for cf_batch, dmat_batch in zip(cor[0],cor[1]):
                var_cf = cf_batch.replace(".batch","").replace(".","_")
                if args.dmat_file is None and 'raw' not in b.outdir and 'true' not in b.outdir:
                    var_dmat = dmat_batch.replace(".batch","").replace(".","_")
                    text += 'afterok:${},afterok:${}'.format(var_cf, var_dmat)
                    if num < len(b.cf):
                        text +=','
                    num += 1 ###
                else:
                    text += 'afterok:${}'.format(var_cf)
                    if num < len(cor[0]):
                        text +=','
                    num += 1 ###
            text += ' {})\n'.format(b.outdir+'/'+stack)
            fout.write(text)
            fout.write('echo "{0}: "${0} \n'.format(var_stack))
        
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

parser.add_argument('--dla-vac',type=str,default=None,required=False,
        help='DLA catalog file')

parser.add_argument('--dmat-file',type=str,default=None,required=False,
        help='use previously calculated distortion matrix')

parser.add_argument("--shuffle-seed", type=int, default = 0,
        required=False, help="seed for shuffled correlation")

parser.add_argument("--use-constant-weight",
        action="store_true", default=False,
        help="Run picca_deltas with eta = fudge = 0, var_lss = 1")

args = parser.parse_args()

if not args.no_deltas:
    try:
        os.makedirs(args.out_dir+"/deltas")
    except FileExistsError:
        pass
try:
    os.makedirs(args.out_dir+"/logs")
except FileExistsError:
    pass

b = batch()
b.outdir = args.out_dir

if "xcf" in args.to_do and args.shuffle_seed == 0 :
    print("Seed for shuffled correlation is 0 by default")


time_debug = "00:10:00"
if "cf" in args.to_do:
    if len(args.zint) > 1:
        time = "01:00:00"
    else:
        time = "02:00:00"
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
    time = "00:30:00"
    if args.debug:
        time = time_debug
    xcf(b,time, args.drq, args.zint, args.out_dir,
            email=args.email,fidOm=args.fid_Om, fidPk=args.fid_Pk, fidOr=args.fid_Or, shuffle_seed = args.shuffle_seed)
    
    time = "01:00:00"
    if args.debug:
        time = time_debug
    xdmat(b,time, args.drq, args.zint, args.out_dir,
            email=args.email, fidOm=args.fid_Om, fidOr=args.fid_Or)

time = "02:00:00"
if args.debug:
    time = time_debug
elif 'raw' in b.outdir:
    time = "00:25:00"
if not args.no_deltas:
    picca_deltas(b,time,args.in_dir, args.out_dir,args.drq,
            email=args.email, debug=args.debug, mode=args.mode,
        lambda_rest_min=args.lambda_rest_min, lambda_rest_max=args.lambda_rest_max, mask_dla_cat=args.dla_vac, use_constant_weight=args.use_constant_weight)

submit(b)

#!/usr/bin/env python
"""Compute the metal matrices
"""
import sys
import os
import time
import argparse
import numpy as np
import fitsio

from picca import constants, cf, io
from picca.utils import userprint

DEFAULT_SI_METALS = ['SiIII(1207)','SiII(1190)','SiII(1193)','SiII(1260)']

def read_stack_deltas_table(filename):
    """
    Read stack.

    Args:
        filename : std , path
    Returns:
        table as numpy.ndarray
    """
    return fitsio.read(filename, "STACK_DELTAS")


def calc_fast_metal_dmat(in_lambda_abs_1,
                         in_lambda_abs_2,
                         out_lambda_abs_1,
                         out_lambda_abs_2,
                         stack_table_1,
                         stack_table_2,
                         rebin_factor=None):
    """Computes the metal distortion matrix.

    Args:
        in_lambda_abs_1 : str
            Name of absorption in picca.constants in forest pixels from stack 1
            (input, i.e. 'true' absorber)
        in_lambda_abs_2 : str
            Name of absorption in picca.constants in forest pixels from stack 2
            (input, i.e. 'true' absorber)
        out_lambda_abs_1 : str
            Name of absorption in picca.constants in forest pixels from stack 1
            (output, i.e. 'assumed' absorber, usually LYA)
        out_lambda_abs_2 : str
            Name of absorption in picca.constants in forest pixels from stack 2
            (output, i.e. 'assumed' absorber, usually LYA)
        stack_table_1: table
            table with cols LOGLAM and WEIGHT for first series of deltas
        stack_table_2: table
            table with cols LOGLAM and WEIGHT for second series of deltas
    Optionnal : rebin_factor
            rebin loglam and weights
    Returns:
        The distortion matrix data
        Note the global picca.cf contains the cosmology and the rp grid
    """

    # stack_table_1 and _2 are the results of the function
    # picca.delta_extraction.expected_flux.get_stack_delta,
    # called in picca.delta_extraction.expected_flux.hdu_stack_deltas
    # It is the stack of weights =
    # picca.delta_extraction.expected_flux.compute_forest_weights(forest, forest.continuum).
    # This function is implemented dr16_expected_flux.Dr16ExpectedFlux.compute_forest_weights
    # and contains the terms  eta * var_pipe + self.var_lss_mod*var_lss + fudge / var_pipe
    # some being set at 0 or 1 depending on the configuration.
    # It does NOT contain the term of redshift evolution (1+z)^(apha-1).

    loglam1 = stack_table_1["LOGLAM"]
    weight1 = stack_table_1["WEIGHT"]
    loglam2 = stack_table_2["LOGLAM"]
    weight2 = stack_table_2["WEIGHT"]
    if rebin_factor is not None:
        size1 = loglam1.size
        loglam1 = loglam1[:(size1 // rebin_factor) * rebin_factor].reshape(
            (size1 // rebin_factor), rebin_factor).mean(-1)
        weight1 = weight1[:(size1 // rebin_factor) * rebin_factor].reshape(
            (size1 // rebin_factor), rebin_factor).mean(-1)
        size2 = loglam2.size
        loglam2 = loglam2[:(size2 // rebin_factor) * rebin_factor].reshape(
            (size2 // rebin_factor), rebin_factor).mean(-1)
        weight2 = weight2[:(size2 // rebin_factor) * rebin_factor].reshape(
            (size2 // rebin_factor), rebin_factor).mean(-1)

    # input
    input_z1 = (10**loglam1) / constants.ABSORBER_IGM[in_lambda_abs_1] - 1.
    input_z2 = (10**loglam2) / constants.ABSORBER_IGM[in_lambda_abs_2] - 1.
    input_r1 = cf.cosmo.get_r_comov(input_z1)
    input_r2 = cf.cosmo.get_r_comov(input_z2)
    # all pairs
    input_rp = (
        input_r1[:, None] -
        input_r2[None, :]).ravel()  # same sign as line 676 of cf.py (1-2)
    if not cf.x_correlation:
        input_rp = np.abs(input_rp)
    input_dist  = ((input_r1[:, None]+input_r2[None, :])/2).ravel()

    # output
    output_z1 = (10**loglam1) / constants.ABSORBER_IGM[out_lambda_abs_1] - 1.
    output_z2 = (10**loglam2) / constants.ABSORBER_IGM[out_lambda_abs_2] - 1.
    output_r1 = cf.cosmo.get_r_comov(output_z1)
    output_r2 = cf.cosmo.get_r_comov(output_z2)
    # all pairs
    output_rp = (
        output_r1[:, None] -
        output_r2[None, :]).ravel()  # same sign as line 676 of cf.py (1-2)
    if not cf.x_correlation:
        output_rp = np.abs(output_rp)
    output_dist = ((output_r1[:, None]+output_r2[None, :])/2).ravel()

    # weights
    # alpha_in: in (1+z)^(alpha_in-1) is a scaling used to model how the metal contribution
    # evolves with redshift (by default alpha_in=1 so that this has no effect)
    # alpha_out: (1+z)^(alpha_out-1) is applied to the delta weights in io.read_deltas and
    # used for the correlation function. It also has to be applied here.
    # we have them for both forests (1 and 2)
    alpha_in_1 = cf.alpha_abs[in_lambda_abs_1]
    alpha_out_1 = cf.alpha  # = args.z_evol in main function
    alpha_in_2 = cf.alpha_abs[in_lambda_abs_2]
    alpha_out_2 = cf.alpha2  # = args.z_evol2 in main function
    # so here we have to apply both scalings (in the original code :
    # alpha_in is applied in cf.calc_metal_dmat and alpha_out in io.read_deltas)
    weights = (
        (weight1 * ((1 + input_z1)**(alpha_in_1 + alpha_out_1 - 2)))[:, None] *
        (weight2 *
         ((1 + input_z2)**(alpha_in_2 + alpha_out_2 - 2)))[None, :]).ravel()

    # r_par distortion matrix
    rpbins   = cf.r_par_min \
        + (cf.r_par_max-cf.r_par_min)/cf.num_bins_r_par*np.arange(cf.num_bins_r_par+1)

    # I checked the orientation of the matrix
    rp_1d_dmat, _, _ = np.histogram2d(output_rp,
                                      input_rp,
                                      bins=(rpbins, rpbins),
                                      weights=weights)
    # normalize
    sum_rp_1d_dmat = np.sum(rp_1d_dmat,axis=0)
    rp_1d_dmat    /= (sum_rp_1d_dmat+(sum_rp_1d_dmat==0))

    # independently, we compute the r_trans distortion matrix
    rtbins   = cf.r_trans_max/cf.num_bins_r_trans*np.arange(cf.num_bins_r_trans+1)
    # we have input_dist , output_dist and weight.
    # we don't need to store the absolute comoving distances but the ratio between output and input.
    # we rebin that to compute the rest faster.
    # histogram of distance scaling with proper weights:
    # dist*theta = r_trans
    # theta_max  = r_trans_max/dist
    # solid angle contibuting for each distance propto theta_max**2 = (r_trans_max/dist)**2 propto 1/dist**2
    # we weight the distances with this additional factor
    # using the input or the output distance in the solid angle weight gives virtually the same result
    #distance_ratio_weights,distance_ratio_bins = np.histogram(output_dist/input_dist,bins=4*rtbins.size,weights=weights/input_dist**2*(input_rp<cf.r_par_max)*(input_rp>cf.r_par_min))
    # we also select only distance ratio for which the input_rp (that of the true separation of the absorbers) is small, so that this
    # fast matrix calculation is accurate where it matters the most
    distance_ratio_weights,distance_ratio_bins = np.histogram(output_dist/input_dist,bins=4*rtbins.size,weights=weights/input_dist**2*(np.abs(input_rp)<20.))
    distance_ratios=(distance_ratio_bins[1:]+distance_ratio_bins[:-1])/2.

    # now we need to scan as a function of separation angles, or equivalently rt.
    rt_bin_centers = (rtbins[:-1]+rtbins[1:])/2.
    rt_bin_half_size = (rtbins[1]-rtbins[0])/2.
    # we are oversampling the correlation function rt grid to correctly compute bin migration.
    oversample  = 7
    delta_rt    = np.linspace(-rt_bin_half_size,rt_bin_half_size*(1-2./oversample),oversample)[None,:] # the -2/oversample term is needed to get a even-spaced grid
    rt_1d_dmat  = np.zeros((cf.num_bins_r_trans,cf.num_bins_r_trans))
    for i,rt in enumerate(rt_bin_centers) :
        # the weight is proportional to rt+delta_rt to get the correct solid angle effect inside the bin (but it's almost a negligible effect)
        rt_1d_dmat[:,i],_ = np.histogram((distance_ratios[:,None]*(rt+delta_rt)[None,:]).ravel(),bins=rtbins,weights=(distance_ratio_weights[:,None]*(rt+delta_rt)[None,:]).ravel())

    # normalize
    sum_rt_1d_dmat = np.sum(rt_1d_dmat,axis=0)
    rt_1d_dmat    /= (sum_rt_1d_dmat+(sum_rt_1d_dmat==0))

    # now that we have both distortion along r_par and r_trans, we have to combine them
    # we just multiply the two matrices, with indices splitted for rt and rp
    # full_index = rt_index + cf.num_bins_r_trans * rp_index
    # rt_index   = full_index%cf.num_bins_r_trans
    # rp_index  = full_index//cf.num_bins_r_trans
    num_bins_total = cf.num_bins_r_par * cf.num_bins_r_trans
    dmat = np.zeros((num_bins_total,num_bins_total))
    pp   = np.arange(num_bins_total,dtype=int)
    for k in range(num_bins_total) :
        dmat[k,pp] = rt_1d_dmat[k%cf.num_bins_r_trans,pp%cf.num_bins_r_trans] * rp_1d_dmat[k//cf.num_bins_r_trans,pp//cf.num_bins_r_trans]

    # compute effective z,rp,rt
    sum_out_weight, _    = np.histogram(output_rp, bins=rpbins, weights=weights)
    sum_out_weight_rp, _ = np.histogram(output_rp,
                                        bins=rpbins,
                                        weights=weights *
                                        (output_rp[None, :].ravel()))

    # return the redshift of the actual absorber, which is the average of input_z1
    # and input_z2
    sum_out_weight_z, _ = np.histogram(
        output_rp,
        bins=rpbins,
        weights=weights *
        (((input_z1[:, None] + input_z2[None, :]) / 2.).ravel()))

    r_par_eff_1d = sum_out_weight_rp / (sum_out_weight + (sum_out_weight == 0))
    z_eff_1d     = sum_out_weight_z / (sum_out_weight + (sum_out_weight == 0))

    # r_trans has no weights here
    r1 = np.arange(cf.num_bins_r_trans)     * cf.r_trans_max / cf.num_bins_r_trans
    r2 = (1+np.arange(cf.num_bins_r_trans)) * cf.r_trans_max / cf.num_bins_r_trans
    r_trans_eff_1d = (2*(r2**3-r1**3))/(3*(r2**2-r1**2)) # this is to account for the solid angle effect on the mean

    full_index = np.arange(num_bins_total)
    rt_index   = full_index%cf.num_bins_r_trans
    rp_index   = full_index//cf.num_bins_r_trans

    r_par_eff_2d   = r_par_eff_1d[rp_index]
    z_eff_2d       = z_eff_1d[rp_index]
    r_trans_eff_2d = r_trans_eff_1d[rt_index]

    return dmat, r_par_eff_2d, r_trans_eff_2d, z_eff_2d


def main(cmdargs):
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Compute the auto and cross-correlation of delta fields for a list of IGM
    absorption."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Computes metal matrices')

    parser.add_argument('--out',
                        type=str,
                        default=None,
                        required=True,
                        help='Output file name')

    parser.add_argument(
        '-i',
        '--in-attributes',
        type=str,
        default=None,
        required=True,
        help='Path to delta_attributes.fits.gz file with hdu STACK_DELTAS'
        ' containing table with at least rows "LOGLAM" and "WEIGHT"')

    parser.add_argument('--in-attributes2',
                        type=str,
                        default=None,
                        required=False,
                        help='Path to 2nd delta_attributes.fits.gz file')

    parser.add_argument(
        '--delta-dir',
        type=str,
        default=None,
        required=False,
        help='Path to directory with delta*.gz to get the blinding info'
        ' (default is trying to guess from attributes file)')

    parser.add_argument('--rp-min',
                        type=float,
                        default=0.,
                        required=False,
                        help='Min r-parallel [h^-1 Mpc]')

    parser.add_argument('--rp-max',
                        type=float,
                        default=200.,
                        required=False,
                        help='Max r-parallel [h^-1 Mpc]')

    parser.add_argument('--rt-max',
                        type=float,
                        default=200.,
                        required=False,
                        help='Max r-transverse [h^-1 Mpc]')

    parser.add_argument('--np',
                        type=int,
                        default=50,
                        required=False,
                        help='Number of r-parallel bins')

    parser.add_argument('--nt',
                        type=int,
                        default=50,
                        required=False,
                        help='Number of r-transverse bins')

    parser.add_argument(
        '--coef-binning-model',
        type=int,
        default=1,
        required=False,
        help=('Coefficient multiplying np and nt to get finner binning for the '
              'model'))

    parser.add_argument(
        '--z-cut-min',
        type=float,
        default=0.,
        required=False,
        help=('Use only pairs of forest x object with the mean of the last '
              'absorber redshift and the object redshift larger than '
              'z-cut-min'))

    parser.add_argument(
        '--z-cut-max',
        type=float,
        default=10.,
        required=False,
        help=('Use only pairs of forest x object with the mean of the last '
              'absorber redshift and the object redshift smaller than '
              'z-cut-max'))

    parser.add_argument('--z-min-sources',
                        type=float,
                        default=0.,
                        required=False,
                        help=('Limit the minimum redshift of the quasars '
                              'used as sources for spectra'))

    parser.add_argument('--z-max-sources',
                        type=float,
                        default=10.,
                        required=False,
                        help=('Limit the maximum redshift of the quasars '
                              'used as sources for spectra'))

    parser.add_argument(
        '--lambda-abs',
        type=str,
        default='LYA',
        required=False,
        help=('Name of the absorption in picca.constants defining the redshift '
              'of the delta'))

    parser.add_argument(
        '--lambda-abs2',
        type=str,
        default=None,
        required=False,
        help=('Name of the absorption in picca.constants defining the redshift '
              'of the 2nd delta'))

    parser.add_argument(
        '--abs-igm',
        type=str,
        default=[],
        required=True,
        nargs='*',
        help=('List of names of metal absorption in picca.constants present in '
              'forest'))

    parser.add_argument(
        '--abs-igm2',
        type=str,
        default=[],
        required=False,
        nargs='*',
        help=('List of names of metal absorption in picca.constants present in '
              '2nd forest'))

    parser.add_argument('--z-ref',
                        type=float,
                        default=2.25,
                        required=False,
                        help='Reference redshift')

    parser.add_argument(
        '--z-evol',
        type=float,
        default=2.9,
        required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument(
        '--z-evol2',
        type=float,
        default=2.9,
        required=False,
        help='Exponent of the redshift evolution of the 2nd delta field')

    parser.add_argument(
        '--metal-alpha',
        type=float,
        default=1.,
        required=False,
        help='Exponent of the redshift evolution of the metal delta field')

    parser.add_argument(
        '--fid-Om',
        type=float,
        default=0.315,
        required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--fid-Or',
        type=float,
        default=0.,
        required=False,
        help='Omega_radiation(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--fid-Ok',
                        type=float,
                        default=0.,
                        required=False,
                        help='Omega_k(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--fid-wl',
        type=float,
        default=-1.,
        required=False,
        help='Equation of state of dark energy of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--unfold-cf',
        action='store_true',
        required=False,
        help='rp can be positive or negative depending on the relative '
        'position between absorber1 and absorber2')

    parser.add_argument(
        '--rebin-factor',
        type=int,
        default=None,
        required=False,
        help='Rebin factor for deltas. If not None, deltas will '
        'be rebinned by that factor')

    parser.add_argument('--relevant-metals',
                    action='store_true',
                    required=False,
                    help='compute only the metal correlations used by Vega'
                       'i.e. 4 LyaxSi matrices and CIVxCIV')

    args = parser.parse_args(cmdargs)

    # setup variables in module cf
    cf.r_par_max = args.rp_max
    cf.r_trans_max = args.rt_max
    cf.r_par_min = args.rp_min
    cf.z_cut_max = args.z_cut_max
    cf.z_cut_min = args.z_cut_min
    cf.num_bins_r_par = args.np * args.coef_binning_model
    cf.num_bins_r_trans = args.nt * args.coef_binning_model
    cf.num_model_bins_r_par = args.np * args.coef_binning_model
    cf.num_model_bins_r_trans = args.nt * args.coef_binning_model
    cf.z_ref = args.z_ref
    cf.alpha = args.z_evol
    cf.alpha2 = args.z_evol2
    cf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    cf.x_correlation = False  # I guess I have to specify this!

    cf.alpha_abs = {}
    cf.alpha_abs[args.lambda_abs] = cf.alpha
    for metal in args.abs_igm:
        cf.alpha_abs[metal] = args.metal_alpha

    # read blinding keyword
    if args.delta_dir is None:
        args.delta_dir = os.path.dirname(args.in_attributes) + "/../Delta/"
        if not os.path.isdir(args.delta_dir):
            userprint(
                "Tried to guess the delta directory (containing the delta*.gz "
                f"files) but '{args.delta_dir}' is not valid"
            )
            userprint("Please specify the directory with option --delta-dir")
            sys.exit(1)
    blinding = io.read_blinding(args.delta_dir)

    # load fiducial cosmology
    cf.cosmo = constants.Cosmo(Om=args.fid_Om,
                               Or=args.fid_Or,
                               Ok=args.fid_Ok,
                               wl=args.fid_wl,
                               blinding=blinding)

    t0 = time.time()

    ### Read data
    stack_table1 = read_stack_deltas_table(args.in_attributes)

    if args.in_attributes2 is not None:
        stack_table2 = read_stack_deltas_table(args.in_attributes2)
    else:
        stack_table2 = stack_table1  # reference to first one

    t1 = time.time()
    userprint(
        f'picca_fast_metal_dmat.py - Time reading data: {(t1-t0)/60:.3f} minutes'
    )

    abs_igm = [args.lambda_abs] + args.abs_igm
    userprint(f"abs_igm = {abs_igm}")

    if args.lambda_abs2 is None:
        args.lambda_abs2 = args.lambda_abs
        args.abs_igm2 = args.abs_igm

    abs_igm_2 = [args.lambda_abs2] + args.abs_igm2

    if cf.x_correlation:
        userprint(f"abs_igm2 = {abs_igm_2}")

    # intitialize arrays to store the results for the different metal absorption
    dmat_all = []
    r_par_all = []
    r_trans_all = []
    z_all = []
    names = []

    # loop over metals
    for index1, abs_igm1 in enumerate(abs_igm):
        index0 = index1
        if args.lambda_abs != args.lambda_abs2:
            index0 = 0
        for index2, abs_igm2 in enumerate(abs_igm_2[index0:]):
            if index1 == 0 and index2 == 0:
                continue

            if args.relevant_metals:
                if not (abs_igm1 == "LYA" and abs_igm2 in DEFAULT_SI_METALS) \
                and not (abs_igm1 == "CIV(eff)" and abs_igm1 == abs_igm2):
                    continue

            userprint("Computing", abs_igm1, abs_igm2)

            # this a matrix as a function of rp only
            dmat, r_par_eff, r_trans_eff, z_eff = calc_fast_metal_dmat(
                abs_igm1,
                abs_igm2,
                args.lambda_abs,
                args.lambda_abs2,
                stack_table1,
                stack_table2,
                rebin_factor=args.rebin_factor)

            # add these results to the list ofor the different metal absorption
            dmat_all.append(dmat)
            r_par_all.append(r_par_eff)
            r_trans_all.append(r_trans_eff)
            z_all.append(z_eff)
            names.append(abs_igm1 + "_" + abs_igm2)

    t2 = time.time()
    userprint(
        f'picca_fast_metal_dmat.py - Time computing all metal matrices : {(t2-t1)/60:.3f} minutes'
    )

    # save the results
    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [{
        'name': 'RPMIN',
        'value': cf.r_par_min,
        'comment': 'Minimum r-parallel [h^-1 Mpc]'
    }, {
        'name': 'RPMAX',
        'value': cf.r_par_max,
        'comment': 'Maximum r-parallel [h^-1 Mpc]'
    }, {
        'name': 'RTMAX',
        'value': cf.r_trans_max,
        'comment': 'Maximum r-transverse [h^-1 Mpc]'
    }, {
        'name': 'NP',
        'value': cf.num_bins_r_par,
        'comment': 'Number of bins in r-parallel'
    }, {
        'name': 'NT',
        'value': cf.num_bins_r_trans,
        'comment': ' Number of bins in r-transverse'
    }, {
        'name': 'COEFMOD',
        'value': args.coef_binning_model,
        'comment': 'Coefficient for model binning'
    }, {
        'name': 'ZCUTMIN',
        'value': cf.z_cut_min,
        'comment': 'Minimum redshift of pairs'
    }, {
        'name': 'ZCUTMAX',
        'value': cf.z_cut_max,
        'comment': 'Maximum redshift of pairs'
    }, {
        'name': 'REJ',
        'value': cf.reject,
        'comment': 'Rejection factor'
    }, {
        'name': 'ALPHAMET',
        'value': args.metal_alpha,
        'comment': 'Evolution of metal bias'
    }, {
        'name': 'OMEGAM',
        'value': args.fid_Om,
        'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'OMEGAR',
        'value': args.fid_Or,
        'comment': 'Omega_radiation(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'OMEGAK',
        'value': args.fid_Ok,
        'comment': 'Omega_k(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name':
            'WL',
        'value':
            args.fid_wl,
        'comment':
            'Equation of state of dark energy of fiducial LambdaCDM cosmology'
    }, {
        'name': "BLINDING",
        'value': blinding,
        'comment': 'String specifying the blinding strategy'
    }]
    len_names = np.array([len(name) for name in names]).max()
    names = np.array(names, dtype='S' + str(len_names))
    results.write([np.array(names)],
                  names=['ABS_IGM'],
                  header=header,
                  comment=['Absorption name'],
                  extname='ATTRI')

    dmat_name = "DM_"
    if blinding != "none":
        dmat_name += "BLIND_"
    names = names.astype(str)
    out_list = []
    out_names = []
    out_comment = []
    out_units = []
    for index, name in enumerate(names):
        out_names += ['RP_' + name]
        out_list += [r_par_all[index]]
        out_comment += ['R-parallel']
        out_units += ['h^-1 Mpc']

        out_names += ['RT_' + name]
        out_list += [r_trans_all[index]]
        out_comment += ['R-transverse']
        out_units += ['h^-1 Mpc']

        out_names += ['Z_' + name]
        out_list += [z_all[index]]
        out_comment += ['Redshift']
        out_units += ['']

        out_names += [dmat_name + name]
        out_list += [dmat_all[index]]
        out_comment += ['Distortion matrix']
        out_units += ['']

        #out_names += ['WDM_' + name]
        #out_list += [weights_dmat_all[index]]
        #out_comment += ['Sum of weight']
        #out_units += ['']

    results.write(out_list,
                  names=out_names,
                  comment=out_comment,
                  units=out_units,
                  extname='MDMAT')
    results.close()

    t3 = time.time()
    userprint(
        f'picca_fast_metal_dmat.py - Time total : {(t3-t0)/60:.3f} minutes')


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)

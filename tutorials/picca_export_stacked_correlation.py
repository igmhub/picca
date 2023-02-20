#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.linalg
import fitsio
import argparse
import copy

from picca.utils import smooth_cov, compute_cov

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        required=True,
        nargs="*",
        help="all data files to stack",
    )

    parser.add_argument("--out",
                        type=str,
                        default=None,
                        required=True,
                        help="output file")

    parser.add_argument(
        "--dmat",
        type=str,
        default=None,
        required=False,
        nargs="*",
        help="distorsion matrix file",
    )

    parser.add_argument(
        "--remove-shuffled-correlation",
        type=str,
        default=None,
        required=False,
        nargs="*",
        help="Remove a correlation from shuffling the distribution of los",
    )

    parser.add_argument(
        "--cov",
        type=str,
        default=None,
        required=False,
        help=
        "covariance matrix file (if not provided it will be calculated by subsampling)",
    )

    parser.add_argument(
        "--do-not-smooth-cov",
        action="store_true",
        default=False,
        help="do not smooth the covariance matrix",
    )

    args = parser.parse_args()

    data = {}

    nbData = len(args.data)

    ### Read data
    for i, p in enumerate(args.data):
        h = fitsio.FITS(p)
        head = h[1].read_header()
        nside = head["NSIDE"]
        ntb = head["NT"]
        npb = head["NP"]
        rt_max = head["RTMAX"]
        rp_min = head["RPMIN"]
        rp_max = head["RPMAX"]
        head = h[2].read_header()
        scheme = head["HLPXSCHM"]
        rp = np.array(h[1]["RP"][:])
        rt = np.array(h[1]["RT"][:])
        z = np.array(h[1]["Z"][:])
        nb = np.array(h[1]["NB"][:])
        da = np.array(h[2]["DA"][:])
        we = np.array(h[2]["WE"][:])
        hep = np.array(h[2]["HEALPID"][:])
        data[i] = {
            "RP": rp,
            "RT": rt,
            "Z": z,
            "NB": nb,
            "DA": da,
            "WE": we,
            "HEALPID": hep,
            "NSIDE": nside,
            "HLPXSCHM": scheme,
            "NP": npb,
            "NT": ntb,
            "RTMAX": rt_max,
            "RPMIN": rp_min,
            "RPMAX": rp_max,
        }
        h.close()

    ###
    if not args.remove_shuffled_correlation is None:
        assert len(args.remove_shuffled_correlation) == len(args.data)
        for i, p in enumerate(args.remove_shuffled_correlation):
            th = fitsio.FITS(p)
            da_s = th["COR"]["DA"][:]
            we_s = th["COR"]["WE"][:]
            da_s = (da_s * we_s).sum(axis=1)
            we_s = we_s.sum(axis=1)
            w = we_s > 0.0
            da_s[w] /= we_s[w]
            th.close()
            data[i]["DA"] -= da_s[:, None]

    ### same header
    for i in range(nbData):
        for k in ["NSIDE", "HLPXSCHM", "NP", "NT", "RTMAX", "RPMIN", "RPMAX"]:
            assert data[i][k] == data[0][k]

    ### Add unshared healpix as empty data
    for i in range(nbData):
        for j in range(nbData):
            w = np.logical_not(np.in1d(data[j]["HEALPID"], data[i]["HEALPID"]))
            if w.sum() > 0:
                new_healpix = data[j]["HEALPID"][w]
                nb_new_healpix = new_healpix.size
                nb_bins = data[i]["DA"].shape[1]
                print("Some healpix are unshared in data {} vs. {}: {}".format(
                    i, j, new_healpix))
                data[i]["DA"] = np.append(data[i]["DA"],
                                          np.zeros((nb_new_healpix, nb_bins)),
                                          axis=0)
                data[i]["WE"] = np.append(data[i]["WE"],
                                          np.zeros((nb_new_healpix, nb_bins)),
                                          axis=0)
                data[i]["HEALPID"] = np.append(data[i]["HEALPID"], new_healpix)

    ### Sort the data by the healpix values
    for i in range(nbData):
        sort = np.array(data[i]["HEALPID"]).argsort()
        data[i]["DA"] = data[i]["DA"][sort]
        data[i]["WE"] = data[i]["WE"][sort]
        data[i]["HEALPID"] = data[i]["HEALPID"][sort]

    ### Sum everything
    final = copy.deepcopy(data[0])
    for i in range(1, nbData):
        we_final = final["WE"].sum(axis=0)
        we_datai = data[i]["WE"].sum(axis=0)
        for k in ["RP", "RT", "Z"]:
            final[k] = final[k] * we_final + data[i][k] * we_datai

        for k in ["DA"]:
            final[k] = final[k] * final["WE"] + data[i][k] * data[i]["WE"]

        for k in ["NB", "WE"]:
            final[k] += data[i][k]

        we_final = final["WE"].sum(axis=0)
        w = we_final > 0.0
        for k in ["RP", "RT", "Z"]:
            final[k][w] /= we_final[w]

        w = final["WE"] > 0.0
        for k in ["DA"]:
            final[k][w] /= final["WE"][w]

    ### Get the covariance matrix
    if args.cov is not None:
        hh = fitsio.FITS(args.cov)
        final["CO"] = hh[1]["CO"][:]
        hh.close()
    else:
        binSizeP = (final["RPMAX"] - final["RPMIN"]) / final["NP"]
        binSizeT = (final["RTMAX"] - 0.0) / final["NT"]
        if not args.do_not_smooth_cov:
            print("INFO: The covariance will be smoothed")
            final["CO"] = smooth_cov(
                final["DA"],
                final["WE"],
                final["RP"],
                final["RT"],
                delta_r_trans=binSizeT,
                delta_r_par=binSizeP,
            )
        else:
            print("INFO: The covariance will not be smoothed")
            final["CO"] = compute_cov(final["DA"], final["WE"])

    ### Test covariance matrix
    try:
        scipy.linalg.cholesky(final["CO"])
    except scipy.linalg.LinAlgError:
        print("WARNING: Matrix is not positive definite")

    ### Measurement
    final["DA"] = (final["DA"] * final["WE"]).sum(axis=0)
    final["WE"] = final["WE"].sum(axis=0)
    w = final["WE"] > 0.0
    final["DA"][w] /= final["WE"][w]

    ### Distortion matrix
    if args.dmat is not None:
        assert len(args.dmat) == len(args.data)
        for i, p in enumerate(args.dmat):
            h = fitsio.FITS(p)
            h2 = fitsio.FITS(args.data[i])
            twdm = h2[2]["WE"][:].sum(axis=0)
            if i == 0:
                dm = (h[1]["DM"][:] * twdm[:, None]).copy()
                wdm = twdm.copy()
            else:
                dm += h[1]["DM"][:] * twdm[:, None]
                wdm += twdm
            h.close()
            h2.close()
        final["DM"] = dm / wdm[:, None]
    else:
        final["DM"] = np.eye(len(final["DA"]))

    h = fitsio.FITS(args.out, "rw", clobber=True)
    head = {}
    for k in ["NT", "NP", "RTMAX", "RPMIN", "RPMAX"]:
        head[k] = final[k]
    names = ["RP", "RT", "Z", "DA", "CO", "DM", "NB"]
    h.write([final[k] for k in names], names=names, header=head, extname="COR")
    h.close()

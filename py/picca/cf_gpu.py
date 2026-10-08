"""GPU (numba.cuda) version of the pair loop in picca.cf.compute_xi.

Same physics and binning as ``cf.compute_xi_forest_pairs_fast``; the only
difference in the output is floating point summation order (atomic adds).

Layout: all forests are flattened once into contiguous arrays (``FlatForests``).
CPU workers only find neighbours and return (index1, index2, angle) arrays;
the GPU then runs one thread block per forest pair and accumulates into one
histogram slot per healpix.

Not supported (the caller must fall back to the CPU path): ``cf.ang_correlation``.
"""
import math

import numpy as np
from numba import cuda

from . import cf, constants

BLOCK = 128


class FlatForests:
    """Concatenated pixel arrays of a list of forests, plus per-forest data."""

    def __init__(self, forests):
        sizes = np.array([f.z.size for f in forests], dtype=np.int64)
        self.offsets = np.zeros(len(forests) + 1, dtype=np.int64)
        np.cumsum(sizes, out=self.offsets[1:])
        cat = lambda name: np.ascontiguousarray(
            np.concatenate([getattr(f, name) for f in forests]), dtype=np.float64
        )
        self.z = cat("z")
        self.r_comov = cat("r_comov")
        self.dist_m = cat("dist_m")
        self.weights = cat("weights")
        self.delta = cat("delta")
        self.z_qso = np.array([f.z_qso for f in forests], dtype=np.float64)
        for k, f in enumerate(forests):
            f.gpu_idx = k

    def to_device(self):
        return tuple(
            cuda.to_device(a)
            for a in (
                self.offsets, self.z, self.r_comov, self.dist_m,
                self.weights, self.delta, self.z_qso,
            )
        )


def pairs_for_healpixs(healpixs):
    """CPU worker: neighbours of every forest in ``healpixs`` as index arrays.

    Must run after ``cf.data`` (and ``cf.data2``) were flattened so that every
    delta has a ``gpu_idx``. Returns a list of
    (healpix, idx1, idx2, ang, same_half_plate) tuples, one per healpix.
    """
    cf.fill_neighs(healpixs)
    out = []
    for healpix in healpixs:
        idx1, idx2, angs, shp = [], [], [], []
        for delta1 in cf.data[healpix]:
            neigh = delta1.neighbours
            if len(neigh) > 0:
                idx1.append(np.full(len(neigh), delta1.gpu_idx, dtype=np.int32))
                idx2.append(np.fromiter((d.gpu_idx for d in neigh), np.int32, len(neigh)))
                angs.append(np.asarray(delta1.get_angle_between(list(neigh)), dtype=np.float64))
                if cf.remove_same_half_plate_close_pairs:
                    shp.append(np.array([_same_half_plate(delta1, d) for d in neigh], dtype=np.uint8))
                else:
                    shp.append(np.zeros(len(neigh), dtype=np.uint8))
            setattr(delta1, "neighbours", None)
        if idx1:
            out.append((healpix, np.concatenate(idx1), np.concatenate(idx2),
                        np.concatenate(angs), np.concatenate(shp)))
        else:
            e = np.zeros(0)
            out.append((healpix, e.astype(np.int32), e.astype(np.int32), e, e.astype(np.uint8)))
    return out


def _same_half_plate(delta1, delta2):
    if isinstance(delta1.fiberid, str) or isinstance(delta2.fiberid, str):
        raise RuntimeError(
            "Trying to figure out if two spectra come from the same half plate "
            "but combined reobservations were given"
        )
    return (delta1.plate == delta2.plate) and (
        (delta1.fiberid <= 500 and delta2.fiberid <= 500)
        or (delta1.fiberid > 500 and delta2.fiberid > 500)
    )


@cuda.jit
def _xi_kernel(
    pair_i1, pair_i2, pair_ang, pair_shp, slot,
    off1, z1, rc1, dm1, w1, dl1, zq1,
    off2, z2, rc2, dm2, w2, dl2, zq2,
    fparams, iparams,
    out_w, out_xi, out_rp, out_rt, out_z, out_n,
):
    p = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    a = pair_i1[p]
    b = pair_i2[p]
    ang = pair_ang[p]
    shp = pair_shp[p]

    r_par_min = fparams[0]
    r_par_max = fparams[1]
    r_trans_max = fparams[2]
    z_min_pairs = fparams[3]
    z_max_pairs = fparams[4]
    zerr_cut_rad = fparams[5]
    zerr_cut_kms = fparams[6]
    speed_light = fparams[7]
    nb_par = iparams[0]
    nb_trans = iparams[1]
    rmu = iparams[2]
    xcorr = iparams[3]
    has_zmin = iparams[4]
    has_zmax = iparams[5]
    has_zerr = iparams[6]
    rm_shp = iparams[7]

    s1 = off1[a]
    n1 = off1[a + 1] - s1
    s2 = off2[b]
    n2 = off2[b + 1] - s2
    zqso1 = zq1[a]
    zqso2 = zq2[b]
    use_zerr = has_zerr != 0 and ang < zerr_cut_rad
    cos_h = math.cos(ang / 2)
    sin_h = math.sin(ang / 2)
    base = slot * (nb_par * nb_trans)
    bin_par_width = (r_par_max - r_par_min) / nb_par

    total = n1 * n2
    for k in range(tid, total, cuda.blockDim.x):
        i = k // n2
        j = k - i * n2
        wi = w1[s1 + i]
        if wi == 0:
            continue
        wj = w2[s2 + j]
        if wj == 0:
            continue
        zi = z1[s1 + i]
        zj = z2[s2 + j]
        if use_zerr:
            z_qF = 0.5 * (zi + zqso2)
            dv = abs(zi - zqso2) / (1 + z_qF) * speed_light
            if dv < zerr_cut_kms:
                continue
        zz = (zi + zj) / 2
        if (has_zmin != 0 and zz < z_min_pairs) or (has_zmax != 0 and zz > z_max_pairs):
            continue
        if use_zerr:
            z_qF = 0.5 * (zj + zqso1)
            dv = abs(zj - zqso1) / (1 + z_qF) * speed_light
            if dv < zerr_cut_kms:
                continue

        r_par = (rc1[s1 + i] - rc2[s2 + j]) * cos_h
        r_trans = (dm1[s1 + i] + dm2[s2 + j]) * sin_h
        if rmu != 0:
            r_trans = math.sqrt(r_trans * r_trans + r_par * r_par)
            r_par = r_par / r_trans
        if xcorr == 0:
            r_par = abs(r_par)
        if r_par >= r_par_max or r_trans >= r_trans_max or r_par < r_par_min:
            continue

        dw1 = dl1[s1 + i] * wi
        dw2 = dl2[s2 + j] * wj
        dw12 = dw1 * dw2
        w12 = wi * wj

        bpar = math.floor((r_par - r_par_min) / (r_par_max - r_par_min) * nb_par)
        btrans = math.floor(r_trans / r_trans_max * nb_trans)
        bn = int(btrans + nb_trans * bpar)

        if rm_shp != 0 and shp != 0:
            if abs(r_par) < bin_par_width:
                continue

        idx = base + bn
        cuda.atomic.add(out_xi, idx, dw12)
        cuda.atomic.add(out_w, idx, w12)
        cuda.atomic.add(out_rp, idx, r_par * w12)
        cuda.atomic.add(out_rt, idx, r_trans * w12)
        cuda.atomic.add(out_z, idx, zz * w12)
        cuda.atomic.add(out_n, idx, 1)


def _params():
    f = np.array(
        [
            cf.r_par_min,
            cf.r_par_max,
            cf.r_trans_max,
            cf.z_min_pairs if cf.z_min_pairs is not None else 0.0,
            cf.z_max_pairs if cf.z_max_pairs is not None else 0.0,
            cf.zerr_cut_deg * np.pi / 180.0 if cf.zerr_cut_deg is not None else 0.0,
            cf.zerr_cut_kms if cf.zerr_cut_kms is not None else 0.0,
            constants.SPEED_LIGHT,
        ],
        dtype=np.float64,
    )
    i = np.array(
        [
            cf.num_bins_r_par,
            cf.num_bins_r_trans,
            int(bool(cf.rmu_binning)),
            int(bool(cf.x_correlation)),
            int(cf.z_min_pairs is not None),
            int(cf.z_max_pairs is not None),
            int(cf.zerr_cut_deg is not None),
            int(bool(cf.remove_same_half_plate_close_pairs)),
        ],
        dtype=np.int64,
    )
    return f, i


class XiAccumulator:
    """Accumulates the per-healpix correlation function on one GPU."""

    def __init__(self, flat1, flat2, healpix_list, device=0):
        cuda.select_device(device)
        self.nbins = cf.num_bins_r_par * cf.num_bins_r_trans
        self.slot_of = {h: s for s, h in enumerate(healpix_list)}
        n = len(healpix_list) * self.nbins
        self.out = [cuda.device_array(n, np.float64) for _ in range(5)]
        self.out_n = cuda.device_array(n, np.int64)
        for a in self.out:
            a[:] = 0
        self.out_n[:] = 0
        self.f1 = flat1.to_device()
        self.f2 = self.f1 if flat2 is flat1 else flat2.to_device()
        fp, ip = _params()
        self.fparams = cuda.to_device(fp)
        self.iparams = cuda.to_device(ip)

    def add(self, healpix, idx1, idx2, ang, shp):
        npairs = idx1.size
        if npairs == 0:
            return
        d = [cuda.to_device(x) for x in (idx1, idx2, ang, shp)]
        w, xi, rp, rt, z = self.out
        _xi_kernel[npairs, BLOCK](
            *d, self.slot_of[healpix], *self.f1, *self.f2, self.fparams, self.iparams,
            w, xi, rp, rt, z, self.out_n,
        )

    def result(self, healpix_list):
        """Same structure as ``np.array([cf.compute_xi(h) for h in healpixs])``."""
        cuda.synchronize()
        nb = self.nbins
        host = [a.copy_to_host().reshape(-1, nb) for a in self.out]
        n = self.out_n.copy_to_host().reshape(-1, nb)
        w, xi, rp, rt, z = host
        res = []
        for h in healpix_list:
            s = self.slot_of[h]
            ww, xx, pp, tt, zz = w[s].copy(), xi[s].copy(), rp[s].copy(), rt[s].copy(), z[s].copy()
            m = ww > 0
            xx[m] /= ww[m]
            pp[m] /= ww[m]
            tt[m] /= ww[m]
            zz[m] /= ww[m]
            res.append((ww, xx, pp, tt, zz, n[s].copy()))
        return res

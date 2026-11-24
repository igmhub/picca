"""GPU (numba.cuda) version of the pair loop in picca.xcf.compute_xi.

The cross-correlation of forests with quasars is the forest-forest kernel of ``cf_gpu`` with
each quasar as a one-pixel "forest" of delta 1 (so ``delta1 * w1 * w2`` is the same product as
in ``xcf.compute_xi_forest_pairs_fast``), the redshift-error cut applied when the pairs are
found (as ``xcf.fill_neighs`` does) and a strict lower bound on r_par. The only difference in
the output is floating point summation order (atomic adds).

CPU workers find the neighbours of every forest with array operations on the flattened sky
positions (``pairs_for_healpixs``) and return index arrays; ``cf_gpu.XiAccumulator`` runs the
pixel pairs on the GPUs.

Not supported (the caller must fall back to the CPU path): ``xcf.ang_correlation``.
"""
import numpy as np
from numba import cuda

from . import cf_gpu, constants, xcf


class FlatObjects:
    """Quasars as one-pixel forests with delta 1; ``cf_gpu`` kernels take them as the second set."""

    def __init__(self, objs):
        n = len(objs)
        f64 = lambda name: np.array([getattr(o, name) for o in objs], dtype=np.float64)
        self.offsets = np.arange(n + 1, dtype=np.int64)
        self.z = f64("z_qso")
        self.z_qso = self.z
        self.r_comov = f64("r_comov")
        self.dist_m = f64("dist_m")
        self.weights = np.array([np.ravel(o.weights)[0] for o in objs], dtype=np.float64)
        self.delta = np.ones(n)
        # sky positions, for the neighbour search
        self.x_cart = f64("x_cart")
        self.y_cart = f64("y_cart")
        self.z_cart = f64("z_cart")
        self.ra = f64("ra")
        self.dec = f64("dec")
        self.cos_dec = f64("cos_dec")
        self.thingid = np.array([o.thingid for o in objs])
        for k, o in enumerate(objs):
            o.gpu_idx = k

    def to_device(self):
        return tuple(
            cuda.to_device(a)
            for a in (
                self.offsets, self.z, self.r_comov, self.dist_m,
                self.weights, self.delta, self.z_qso,
            )
        )


_FLAT = {}


def set_flat(flat_forests, flat_objects):
    """Make the flattened forests and objects visible to ``pairs_for_healpixs`` (call before forking)."""
    _FLAT["forests"], _FLAT["objs"] = flat_forests, flat_objects
    _FLAT.pop("r_forests", None)


def _ranges(dct):
    # flattened in sorted healpix order, so each healpix is a contiguous range
    return {h: (v[0].gpu_idx, v[-1].gpu_idx + 1) for h, v in dct.items() if len(v)}


def pairs_for_healpixs(healpixs):
    """CPU worker: the (forest, quasar) pairs of the forests in ``healpixs`` as index arrays.

    Same selection as ``xcf.fill_neighs`` (angle < ang_max, different thingid, redshift-error cut,
    and the r_par window of the forest against the quasar's comoving distance), with the same
    expressions; the candidates of a healpix are the quasars in the healpixes within ``ang_max``
    plus one pixel radius of its centre. Returns a list of
    (healpix, idx_forest, idx_obj, ang, same_half_plate) tuples, one per healpix.
    """
    # pylint: disable=import-outside-toplevel
    import healpy

    fl, ob = _FLAT["forests"], _FLAT["objs"]
    if "r_forests" not in _FLAT:
        _FLAT["r_forests"] = _ranges(xcf.data)
        _FLAT["r_objs"] = _ranges(xcf.objs)
        _FLAT["first_r"] = fl.r_comov[fl.offsets[:-1]]
        _FLAT["last_r"] = fl.r_comov[fl.offsets[1:] - 1]
    rng_f, rng_o = _FLAT["r_forests"], _FLAT["r_objs"]
    first_r, last_r = _FLAT["first_r"], _FLAT["last_r"]
    radius = xcf.ang_max + healpy.max_pixrad(xcf.nside)
    cos_cut = np.cos(xcf.ang_max) - 1e-9
    f_win = xcf.r_trans_max if xcf.rmu_binning else 1
    out = []
    for hp in healpixs:
        lo, hi = rng_f[hp]
        near = healpy.query_disc(xcf.nside, healpy.pix2vec(xcf.nside, hp), radius, inclusive=True)
        spans = [rng_o[h] for h in near if h in rng_o]
        cand = np.concatenate([np.arange(a, b) for a, b in spans]) if spans else np.zeros(0, np.int64)
        xc, yc, zc = ob.x_cart[cand], ob.y_cart[cand], ob.z_cart[cand]
        rows = max(1, int(2e7 // max(cand.size, 1)))
        idx1, idx2, angs = [], [], []
        for r0 in range(lo, hi, rows):
            r1 = min(hi, r0 + rows)
            cos = (xc[None, :] * fl.x_cart[r0:r1, None] + yc[None, :] * fl.y_cart[r0:r1, None]
                   + zc[None, :] * fl.z_cart[r0:r1, None])
            ii, jj = np.nonzero(cos > cos_cut)
            if ii.size == 0:
                continue
            i = ii + r0
            j = cand[jj]
            ang = np.arccos(np.clip(cos[ii, jj], -1.0, 1.0))
            small = ((np.absolute(ob.ra[j] - fl.ra[i]) < constants.SMALL_ANGLE_CUT_OFF)
                     & (np.absolute(ob.dec[j] - fl.dec[i]) < constants.SMALL_ANGLE_CUT_OFF))
            if small.any():
                ang[small] = np.sqrt(
                    (ob.dec[j][small] - fl.dec[i][small]) ** 2
                    + (fl.cos_dec[i][small] * (ob.ra[j][small] - fl.ra[i][small])) ** 2
                )
            keep = (ang < xcf.ang_max) & (fl.thingid[i] != ob.thingid[j])
            if xcf.zerr_cut_deg is not None:
                ang_deg = 180.0 / np.pi * ang
                z_qq = 0.5 * (fl.z_qso[i] + ob.z_qso[j])
                dv_kms = np.abs(fl.z_qso[i] - ob.z_qso[j]) / (1 + z_qq)
                dv_kms *= constants.SPEED_LIGHT
                keep &= ~((ang_deg < xcf.zerr_cut_deg) & (dv_kms < xcf.zerr_cut_kms))
            cos_half = np.cos(ang / 2.0)
            keep &= (first_r[i] - ob.r_comov[j]) * cos_half < xcf.r_par_max * f_win
            keep &= (last_r[i] - ob.r_comov[j]) * cos_half > xcf.r_par_min * f_win
            idx1.append(i[keep].astype(np.int32))
            idx2.append(j[keep].astype(np.int32))
            angs.append(ang[keep])
        if idx1:
            idx1, idx2, angs = np.concatenate(idx1), np.concatenate(idx2), np.concatenate(angs)
        else:
            idx1, idx2, angs = np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0)
        out.append((hp, idx1, idx2, angs, np.zeros(idx1.size, dtype=np.uint8)))
    return out

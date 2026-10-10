"""GPU (numba.cuda) version of the pair loop in picca.cf.compute_dmat.

Same physics, binning and pair selection as ``cf.compute_dmat_forest_pairs_fast``;
the only difference in the output is floating point summation order (atomic adds).

Layout: as in ``cf_gpu`` the forests are flattened once. CPU workers find the
neighbours and draw the pair rejection with the same random stream as the CPU code
(seeded per chunk), and return index arrays. The GPU then runs one thread block per
forest pair in five steps (all steps see only that pair):

  1. select the in-range pixel pairs (data bin, model bin, selected flag) into a list;
  2. compact the unique model bins touched by the pair (K of them);
  3. accumulate the "eta" tables (pixel x unique-model-bin) with atomics;
  4. add the identity term and the eta combination to the global matrix, one atomic
     per (selected pixel pair, unique model bin).

Per-block scratch is allocated once (persistent blocks loop over the pairs).
"""
import math

import numpy as np
from numba import cuda, int32

from . import cf, constants
from .cf_gpu import FlatForests as _FlatForests
from .cf_gpu import _same_half_plate

THREADS = 256


class FlatForestsDmat(_FlatForests):
    """``FlatForests`` plus the per-forest quantities the dmat kernel needs."""

    def __init__(self, forests):
        super().__init__(forests)
        llm, sw, swsll, order = [], [], [], []
        with np.errstate(all="ignore"):
            for f in forests:
                w = f.weights
                s = w.sum()
                mean = np.sum(f.log_lambda * w) / s
                d = f.log_lambda - mean
                llm.append(d)
                sw.append(s)
                swsll.append((w * d**2).sum())
                order.append(-1 if f.order is None else int(f.order))
        self.llm = np.ascontiguousarray(np.concatenate(llm), dtype=np.float64)
        self.sum_w = np.array(sw, dtype=np.float64)
        self.swsll = np.array(swsll, dtype=np.float64)
        self.order = np.array(order, dtype=np.int32)
        sizes = np.diff(self.offsets)
        self.max_pixels = int(sizes.max()) if sizes.size else 0

    def to_device_dmat(self):
        return tuple(
            cuda.to_device(a)
            for a in (
                self.offsets, self.z, self.r_comov, self.dist_m, self.weights,
                self.llm, self.z_qso, self.sum_w, self.swsll, self.order,
            )
        )


def pairs_for_chunk(healpixs):
    """CPU worker: neighbours of the forests in ``healpixs`` after the pair rejection.

    Reproduces the random stream of ``picca_dmat.calc_dmat`` (``np.random.seed(healpixs[0])``
    after ``fill_neighs``, one ``rand`` per forest in healpix order), so for the same chunks
    the GPU run uses exactly the same pairs as the CPU run.
    Returns (idx1, idx2, ang, same_half_plate, num_pairs, num_pairs_used).
    """
    cf.fill_neighs(healpixs)
    np.random.seed(healpixs[0])
    idx1, idx2, angs, shp = [], [], [], []
    num_pairs = 0
    num_pairs_used = 0
    for healpix in healpixs:
        for delta1 in cf.data[healpix]:
            if delta1.order is None:
                raise RuntimeError(
                    "Trying to compute the distortion matrix but order is not defined "
                    "for the deltas. Check previous warning to solve this issue"
                )
            neigh = delta1.neighbours
            keep = np.random.rand(len(neigh)) > cf.reject
            num_pairs += len(neigh)
            num_pairs_used += int(keep.sum())
            if keep.any():
                chosen = list(np.array(neigh)[keep])
                for d in chosen:
                    if d.order is None:
                        raise RuntimeError(
                            "Trying to compute the distortion matrix but order is not "
                            "defined for the deltas. Check previous warning to solve this issue"
                        )
                idx1.append(np.full(len(chosen), delta1.gpu_idx, dtype=np.int32))
                idx2.append(np.fromiter((d.gpu_idx for d in chosen), np.int32, len(chosen)))
                angs.append(np.asarray(delta1.get_angle_between(chosen), dtype=np.float64))
                if cf.remove_same_half_plate_close_pairs:
                    shp.append(np.array([_same_half_plate(delta1, d) for d in chosen], dtype=np.uint8))
                else:
                    shp.append(np.zeros(len(chosen), dtype=np.uint8))
            setattr(delta1, "neighbours", None)
    if idx1:
        return (np.concatenate(idx1), np.concatenate(idx2), np.concatenate(angs),
                np.concatenate(shp), num_pairs, num_pairs_used)
    e = np.zeros(0)
    return e.astype(np.int32), e.astype(np.int32), e, e.astype(np.uint8), num_pairs, num_pairs_used


@cuda.jit
def _dmat_kernel(
    pair_i1, pair_i2, pair_ang, pair_shp, npairs,
    off1, z1, rc1, dm1, w1, ll1, zq1, sw1, sl1, ord1,
    off2, z2, rc2, dm2, w2, ll2, zq2, sw2, sl2, ord2,
    fparams, iparams, kcap, pcap,
    s_i, s_j, s_b, s_m, slot_of, ukeys, eta, flags,
    dmat, wdmat, rpar_eff, rtr_eff, z_eff, w_eff,
):
    """Persistent blocks: block b handles pairs b, b + gridDim.x, ...

    Scratch (per block): s_* int32[pcap], slot_of int32[nmodel], ukeys int32[kcap],
    eta float64[(2*max1 + 2*max2 + 4) * kcap] (``eta`` is sliced below).
    ``flags`` int32[2]: [0] = number of pairs that exceeded the scratch capacity.
    """
    tid = cuda.threadIdx.x
    nthr = cuda.blockDim.x
    blk = cuda.blockIdx.x

    sh = cuda.shared.array(2, dtype=int32)  # [0] list length, [1] K

    r_par_min = fparams[0]
    r_par_max = fparams[1]
    r_trans_max = fparams[2]
    z_min_pairs = fparams[3]
    z_max_pairs = fparams[4]
    zerr_cut_rad = fparams[5]
    zerr_cut_kms = fparams[6]
    speed_light = fparams[7]
    z_ref = fparams[8]
    alpha = fparams[9]
    alpha2 = fparams[10]
    nb_par = iparams[0]
    nb_trans = iparams[1]
    nmb_par = iparams[2]
    nmb_trans = iparams[3]
    rmu = iparams[4]
    xcorr = iparams[5]
    has_zmin = iparams[6]
    has_zmax = iparams[7]
    has_zerr = iparams[8]
    rm_shp = iparams[9]
    z_evol = iparams[10]
    max1 = iparams[11]
    max2 = iparams[12]
    nmodel = nmb_par * nmb_trans

    pb = blk * pcap  # scratch bases for this block
    sb = blk * nmodel
    kb = blk * kcap
    eb = blk * ((2 * max1 + 2 * max2 + 4) * kcap)

    for p in range(blk, npairs, cuda.gridDim.x):
        a = pair_i1[p]
        b = pair_i2[p]
        ang = pair_ang[p]
        shp = pair_shp[p]
        s1 = off1[a]
        n1 = off1[a + 1] - s1
        s2 = off2[b]
        n2 = off2[b + 1] - s2
        zqso1 = zq1[a]
        zqso2 = zq2[b]
        sumw1 = sw1[a]
        sumw2 = sw2[b]
        swsll1 = sl1[a]
        swsll2 = sl2[b]
        order1 = ord1[a]
        order2 = ord2[b]
        use_zerr = has_zerr != 0 and ang < zerr_cut_rad
        cos_h = math.cos(ang / 2)
        sin_h = math.sin(ang / 2)
        bin_par_width = (r_par_max - r_par_min) / nb_par

        if tid == 0:
            sh[0] = 0
            sh[1] = 0
        for m in range(tid, nmodel, nthr):
            slot_of[sb + m] = -1
        cuda.syncthreads()

        # 1. in-range pixel pairs
        total = n1 * n2
        for k in range(tid, total, nthr):
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
            r_par = (rc1[s1 + i] - rc2[s2 + j]) * cos_h
            r_trans = (dm1[s1 + i] + dm2[s2 + j]) * sin_h
            if rmu != 0:
                r_trans = math.sqrt(r_trans * r_trans + r_par * r_par)
                r_par = r_par / r_trans
            if xcorr == 0:
                r_par = abs(r_par)
            if r_par >= r_par_max or r_trans >= r_trans_max or r_par < r_par_min:
                continue

            i_sel = True
            j_sel = True
            zz = (zi + zj) / 2
            if use_zerr:
                z_qF = 0.5 * (zi + zqso2)
                dv = abs(zi - zqso2) / (1 + z_qF) * speed_light
                if dv < zerr_cut_kms:
                    i_sel = False
                z_qF = 0.5 * (zj + zqso1)
                dv = abs(zj - zqso1) / (1 + z_qF) * speed_light
                if dv < zerr_cut_kms:
                    j_sel = False
            if (has_zmin != 0 and zz < z_min_pairs) or (has_zmax != 0 and zz > z_max_pairs):
                j_sel = False
            if rm_shp != 0 and shp != 0:
                if abs(r_par) < bin_par_width:
                    j_sel = False

            bpar = math.floor((r_par - r_par_min) / (r_par_max - r_par_min) * nb_par)
            btrans = math.floor(r_trans / r_trans_max * nb_trans)
            bins = int(btrans + nb_trans * bpar)
            mpar = math.floor((r_par - r_par_min) / (r_par_max - r_par_min) * nmb_par)
            mtrans = math.floor(r_trans / r_trans_max * nmb_trans)
            mb = int(mtrans + nmb_trans * mpar)

            pos = cuda.atomic.add(sh, 0, 1)
            if pos < pcap:
                s_i[pb + pos] = i
                s_j[pb + pos] = j
                s_m[pb + pos] = mb
                if i_sel and j_sel:
                    s_b[pb + pos] = bins
                    w12 = wi * wj
                    cuda.atomic.add(wdmat, bins, w12)
                    cuda.atomic.add(rpar_eff, mb, w12 * r_par)
                    cuda.atomic.add(rtr_eff, mb, w12 * r_trans)
                    cuda.atomic.add(z_eff, mb, w12 * zz)
                    cuda.atomic.add(w_eff, mb, w12)
                else:
                    s_b[pb + pos] = -1
                slot_of[sb + mb] = 0
        cuda.syncthreads()

        nlist = sh[0]
        if nlist == 0:
            cuda.syncthreads()
            continue

        # 2. unique model bins -> slots 0..K-1
        for m in range(tid, nmodel, nthr):
            if slot_of[sb + m] == 0:
                s = cuda.atomic.add(sh, 1, 1)
                slot_of[sb + m] = s
                if s < kcap:
                    ukeys[kb + s] = m
        cuda.syncthreads()
        K = sh[1]
        if nlist > pcap or K > kcap:
            if tid == 0:
                cuda.atomic.add(flags, 0, 1)
            cuda.syncthreads()
            continue

        # 3. eta tables (zeroed first); layouts: eta1,eta3 [n1][K], eta2,eta4 [n2][K], eta5..8 [K]
        e1 = eb
        e3 = e1 + n1 * K
        e2 = e3 + n1 * K
        e4 = e2 + n2 * K
        e5 = e4 + n2 * K
        e6 = e5 + K
        e7 = e6 + K
        e8 = e7 + K
        for t in range(tid, e8 + K - eb, nthr):
            eta[eb + t] = 0.0
        cuda.syncthreads()

        for q in range(tid, nlist, nthr):
            i = s_i[pb + q]
            j = s_j[pb + q]
            slot = slot_of[sb + s_m[pb + q]]
            wi = w1[s1 + i]
            wj = w2[s2 + j]
            if z_evol != 0:
                zfac = (((1 + z1[s1 + i]) / (1 + z_ref)) ** (alpha - 1)) * (
                    ((1 + z2[s2 + j]) / (1 + z_ref)) ** (alpha2 - 1)
                )
            else:
                zfac = 1.0
            li = ll1[s1 + i]
            lj = ll2[s2 + j]
            cuda.atomic.add(eta, e1 + i * K + slot, zfac * wj / sumw2)
            cuda.atomic.add(eta, e2 + j * K + slot, zfac * wi / sumw1)
            cuda.atomic.add(eta, e5 + slot, zfac * (wi * wj) / sumw1 / sumw2)
            if order2 == 1:
                cuda.atomic.add(eta, e3 + i * K + slot, zfac * wj * lj / swsll2)
                cuda.atomic.add(eta, e6 + slot, zfac * wi / sumw1 * (wj * lj / swsll2))
            if order1 == 1:
                cuda.atomic.add(eta, e4 + j * K + slot, zfac * wi * li / swsll1)
                cuda.atomic.add(eta, e7 + slot, zfac * wj / sumw2 * (wi * li / swsll1))
                if order2 == 1:
                    cuda.atomic.add(
                        eta, e8 + slot,
                        zfac * wi * li * wj * lj / swsll1 / swsll2,
                    )
        cuda.syncthreads()

        # 4a. identity term, one atomic per selected pixel pair
        for q in range(tid, nlist, nthr):
            bn = s_b[pb + q]
            if bn >= 0:
                i = s_i[pb + q]
                j = s_j[pb + q]
                if z_evol != 0:
                    zfac = (((1 + z1[s1 + i]) / (1 + z_ref)) ** (alpha - 1)) * (
                        ((1 + z2[s2 + j]) / (1 + z_ref)) ** (alpha2 - 1)
                    )
                else:
                    zfac = 1.0
                cuda.atomic.add(dmat, bn * nmodel + s_m[pb + q], w1[s1 + i] * w2[s2 + j] * zfac)

        # 4b. eta combination: one atomic per (selected pixel pair, unique model bin)
        for t in range(tid, nlist * K, nthr):
            q = t // K
            s = t - q * K
            bn = s_b[pb + q]
            if bn < 0:
                continue
            i = s_i[pb + q]
            j = s_j[pb + q]
            li = ll1[s1 + i]
            lj = ll2[s2 + j]
            val = (
                eta[e5 + s]
                + eta[e6 + s] * lj
                + eta[e7 + s] * li
                + eta[e8 + s] * li * lj
                - eta[e1 + i * K + s]
                - eta[e2 + j * K + s]
                - eta[e3 + i * K + s] * lj
                - eta[e4 + j * K + s] * li
            )
            cuda.atomic.add(dmat, bn * nmodel + ukeys[kb + s], w1[s1 + i] * w2[s2 + j] * val)
        cuda.syncthreads()


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
            cf.z_ref,
            cf.alpha,
            cf.alpha2,
        ],
        dtype=np.float64,
    )
    i = np.array(
        [
            cf.num_bins_r_par,
            cf.num_bins_r_trans,
            cf.num_model_bins_r_par,
            cf.num_model_bins_r_trans,
            int(bool(cf.rmu_binning)),
            int(bool(cf.x_correlation)),
            int(cf.z_min_pairs is not None),
            int(cf.z_max_pairs is not None),
            int(cf.zerr_cut_deg is not None),
            int(bool(cf.remove_same_half_plate_close_pairs)),
            int(bool(cf.redshift_evolution_in_distortion_matrix)),
        ],
        dtype=np.int64,
    )
    return f, i


class _DeviceDmat:
    """Accumulates the distortion matrix on one GPU."""

    BLOCKS_PER_SM = 2
    KCAP = 2048  # max unique model bins per forest pair (measured: mean ~510, max ~1350)

    def __init__(self, flat1, flat2, device=0):
        self.device = device
        self.nbins = cf.num_bins_r_par * cf.num_bins_r_trans
        self.nmodel = cf.num_model_bins_r_par * cf.num_model_bins_r_trans
        z = lambda n: cuda.to_device(np.zeros(n, dtype=np.float64))
        self.dmat = z(self.nbins * self.nmodel)
        self.wdmat = z(self.nbins)
        self.rpar, self.rtr, self.zeff, self.weff = (z(self.nmodel) for _ in range(4))
        self.f1 = flat1.to_device_dmat()
        self.f2 = self.f1 if flat2 is flat1 else flat2.to_device_dmat()
        fp, ip = _params()
        max1, max2 = flat1.max_pixels, flat2.max_pixels
        ip = np.concatenate([ip, [max1, max2]]).astype(np.int64)
        self.fparams = cuda.to_device(fp)
        self.iparams = cuda.to_device(ip)
        self.nblocks = cuda.get_current_device().MULTIPROCESSOR_COUNT * self.BLOCKS_PER_SM
        self.pcap = max1 * max2
        nb, kcap = self.nblocks, self.KCAP
        self.scratch = (
            cuda.device_array(nb * self.pcap, np.int32),
            cuda.device_array(nb * self.pcap, np.int32),
            cuda.device_array(nb * self.pcap, np.int32),
            cuda.device_array(nb * self.pcap, np.int32),
            cuda.device_array(nb * self.nmodel, np.int32),
            cuda.device_array(nb * kcap, np.int32),
            cuda.device_array(nb * (2 * max1 + 2 * max2 + 4) * kcap, np.float64),
        )
        self.flags = cuda.to_device(np.zeros(2, dtype=np.int32))

    def add(self, idx1, idx2, ang, shp):
        npairs = idx1.size
        if npairs == 0:
            return
        d = [cuda.to_device(x) for x in (idx1, idx2, ang, shp)]
        _dmat_kernel[self.nblocks, THREADS](
            *d, npairs, *self.f1, *self.f2, self.fparams, self.iparams,
            self.KCAP, self.pcap, *self.scratch, self.flags,
            self.dmat, self.wdmat, self.rpar, self.rtr, self.zeff, self.weff,
        )

    def host_sums(self):
        cuda.synchronize()
        if self.flags.copy_to_host()[0] != 0:
            raise RuntimeError(
                f"dmat_gpu: {self.flags.copy_to_host()[0]} forest pairs exceeded the scratch "
                f"capacity (KCAP={self.KCAP}); raise _DeviceDmat.KCAP"
            )
        return (
            self.wdmat.copy_to_host(), self.dmat.copy_to_host(), self.rpar.copy_to_host(),
            self.rtr.copy_to_host(), self.zeff.copy_to_host(), self.weff.copy_to_host(),
        )


class DmatAccumulator:
    """Spreads chunks round-robin over the GPUs of the node; ``result`` returns the
    same tuple as ``cf.compute_dmat`` (summed over everything that was added)."""

    def __init__(self, flat1, flat2, devices=None):
        if devices is None:
            devices = list(range(len(cuda.gpus)))
        self.parts = []
        for dev in devices:
            with cuda.gpus[dev]:
                self.parts.append(_DeviceDmat(flat1, flat2, device=dev))
        self.next = 0
        self.num_pairs = 0
        self.num_pairs_used = 0

    def add(self, idx1, idx2, ang, shp, num_pairs, num_pairs_used):
        self.num_pairs += num_pairs
        self.num_pairs_used += num_pairs_used
        part = self.parts[self.next % len(self.parts)]
        self.next += 1
        with cuda.gpus[part.device]:
            part.add(idx1, idx2, ang, shp)

    def result(self):
        sums = None
        for part in self.parts:
            with cuda.gpus[part.device]:
                h = part.host_sums()
            sums = list(h) if sums is None else [a + b for a, b in zip(sums, h)]
        wdmat, dmat, rpar, rtr, zeff, weff = sums
        nbins = cf.num_bins_r_par * cf.num_bins_r_trans
        dmat = dmat.reshape(nbins, cf.num_model_bins_r_par * cf.num_model_bins_r_trans)
        return (wdmat, dmat, rpar, rtr, zeff, weff, self.num_pairs, self.num_pairs_used)

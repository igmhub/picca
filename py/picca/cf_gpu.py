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
import queue
import threading

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
        # sky positions, for the vectorised neighbour search (pairs_for_healpixs_fast)
        per_forest = lambda name: np.array([getattr(f, name) for f in forests])
        self.x_cart = per_forest("x_cart")
        self.y_cart = per_forest("y_cart")
        self.z_cart = per_forest("z_cart")
        self.ra = per_forest("ra")
        self.dec = per_forest("dec")
        self.cos_dec = per_forest("cos_dec")
        self.thingid = per_forest("thingid")
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


_FLAT = {}


def set_flat_forests(flat1, flat2):
    """Make the flattened forests visible to ``pairs_for_healpixs_fast`` (call before forking)."""
    _FLAT["f1"], _FLAT["f2"] = flat1, flat2


def _healpix_ranges(dct):
    # forests are flattened in sorted healpix order, so each healpix is a contiguous range
    return {h: (v[0].gpu_idx, v[-1].gpu_idx + 1) for h, v in dct.items() if len(v)}


def pairs_for_healpixs_fast(healpixs):
    """Same pairs as ``pairs_for_healpixs`` (not in the same order), found with array operations
    on the flattened sky positions instead of one ``query_disc`` and Python lists per forest.

    The candidates of a healpix are the forests in the healpixes within ``ang_max`` plus one
    pixel radius of its centre; the selection (``angle < ang_max``, different thingid, and
    ``ra`` ordering for the auto-correlation) uses the same expressions as ``cf.fill_neighs``.
    Does not handle ``remove_same_half_plate_close_pairs``.
    """
    # pylint: disable=import-outside-toplevel
    import healpy

    f1 = _FLAT["f1"]
    auto = cf.data2 is None
    f2 = f1 if auto else _FLAT["f2"]
    if "r1" not in _FLAT:
        _FLAT["r1"] = _healpix_ranges(cf.data)
        _FLAT["r2"] = _FLAT["r1"] if auto else _healpix_ranges(cf.data2)
    rng1, rng2 = _FLAT["r1"], _FLAT["r2"]
    radius = cf.ang_max + healpy.max_pixrad(cf.nside)
    cos_cut = np.cos(cf.ang_max) - 1e-9
    out = []
    for hp in healpixs:
        lo, hi = rng1[hp]
        near = healpy.query_disc(cf.nside, healpy.pix2vec(cf.nside, hp), radius, inclusive=True)
        spans = [rng2[h] for h in near if h in rng2]
        cand = np.concatenate([np.arange(a, b) for a, b in spans]) if spans else np.zeros(0, np.int64)
        xc, yc, zc = f2.x_cart[cand], f2.y_cart[cand], f2.z_cart[cand]
        rows = max(1, int(2e7 // max(cand.size, 1)))
        idx1, idx2, angs = [], [], []
        for r0 in range(lo, hi, rows):
            r1 = min(hi, r0 + rows)
            cos = (xc[None, :] * f1.x_cart[r0:r1, None] + yc[None, :] * f1.y_cart[r0:r1, None]
                   + zc[None, :] * f1.z_cart[r0:r1, None])
            ii, jj = np.nonzero(cos > cos_cut)
            if ii.size == 0:
                continue
            i = ii + r0
            j = cand[jj]
            c = np.clip(cos[ii, jj], -1.0, 1.0)
            ang = np.arccos(c)
            small = ((np.absolute(f2.ra[j] - f1.ra[i]) < constants.SMALL_ANGLE_CUT_OFF)
                     & (np.absolute(f2.dec[j] - f1.dec[i]) < constants.SMALL_ANGLE_CUT_OFF))
            if small.any():
                ang[small] = np.sqrt(
                    (f2.dec[j][small] - f1.dec[i][small]) ** 2
                    + (f1.cos_dec[i][small] * (f2.ra[j][small] - f1.ra[i][small])) ** 2
                )
            keep = (ang < cf.ang_max) & (f1.thingid[i] != f2.thingid[j])
            if auto:
                keep &= f1.ra[i] > f2.ra[j]
            idx1.append(i[keep].astype(np.int32))
            idx2.append(j[keep].astype(np.int32))
            angs.append(ang[keep])
        if idx1:
            idx1, idx2, angs = np.concatenate(idx1), np.concatenate(idx2), np.concatenate(angs)
        else:
            idx1, idx2, angs = np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0)
        out.append((hp, idx1, idx2, angs, np.zeros(idx1.size, dtype=np.uint8)))
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


@cuda.jit(device=True)
def _accumulate_pair(
    p, pair_i1, pair_i2, pair_ang, pair_shp,
    off1, z1, rc1, dm1, w1, dl1, zq1,
    off2, z2, rc2, dm2, w2, dl2, zq2,
    fparams, iparams, acc, base, nbins,
):
    """Accumulate one forest pair with the calling block's threads.

    ``acc`` holds six planes of ``nbins`` float64: xi, weight, r_par, r_trans,
    z (all weighted sums) and the pair count; ``base`` is the offset of plane 0.
    """
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

        dw12 = (dl1[s1 + i] * wi) * (dl2[s2 + j] * wj)
        w12 = wi * wj

        bpar = math.floor((r_par - r_par_min) / (r_par_max - r_par_min) * nb_par)
        btrans = math.floor(r_trans / r_trans_max * nb_trans)
        bn = int(btrans + nb_trans * bpar)

        if rm_shp != 0 and shp != 0:
            if abs(r_par) < bin_par_width:
                continue

        idx = base + bn
        cuda.atomic.add(acc, idx, dw12)
        cuda.atomic.add(acc, idx + nbins, w12)
        cuda.atomic.add(acc, idx + 2 * nbins, r_par * w12)
        cuda.atomic.add(acc, idx + 3 * nbins, r_trans * w12)
        cuda.atomic.add(acc, idx + 4 * nbins, zz * w12)
        cuda.atomic.add(acc, idx + 5 * nbins, 1.0)


@cuda.jit
def _xi_kernel_global(
    pair_i1, pair_i2, pair_ang, pair_shp, pair_slot, npairs, pairs_per_block,
    off1, z1, rc1, dm1, w1, dl1, zq1,
    off2, z2, rc2, dm2, w2, dl2, zq2,
    fparams, iparams, nbins, out,
):
    """Fallback: accumulate straight into global memory (slow, ~8x)."""
    p0 = cuda.blockIdx.x * pairs_per_block
    p1 = p0 + pairs_per_block
    if p1 > npairs:
        p1 = npairs
    for p in range(p0, p1):
        _accumulate_pair(
            p, pair_i1, pair_i2, pair_ang, pair_shp,
            off1, z1, rc1, dm1, w1, dl1, zq1,
            off2, z2, rc2, dm2, w2, dl2, zq2,
            fparams, iparams, out, pair_slot[p] * 6 * nbins, nbins,
        )


@cuda.jit(max_registers=int(__import__('os').environ.get('PICCA_GPU_MAXREG', 64)))
def _xi_kernel_shared(
    pair_i1, pair_i2, pair_ang, pair_shp, pair_slot, npairs, pairs_per_block,
    off1, z1, rc1, dm1, w1, dl1, zq1,
    off2, z2, rc2, dm2, w2, dl2, zq2,
    fparams, iparams, nbins, out,
):
    """Each block accumulates ``pairs_per_block`` consecutive pairs into a shared-memory
    histogram (6 planes x nbins float64, dynamic shared memory) and flushes it to the output
    slot of those pairs (``pair_slot``, constant over runs of pairs; the block flushes and
    restarts when it changes). ``nbins`` = 0 makes this a no-op (used to compile)."""
    sh = cuda.shared.array(0, dtype=np.float64)
    tid = cuda.threadIdx.x
    nthr = cuda.blockDim.x
    p0 = cuda.blockIdx.x * pairs_per_block
    p1 = p0 + pairs_per_block
    if p1 > npairs:
        p1 = npairs
    if p0 >= p1:
        return
    for t in range(tid, 6 * nbins, nthr):
        sh[t] = 0.0
    cuda.syncthreads()
    cur = pair_slot[p0]
    for p in range(p0, p1):
        slot = pair_slot[p]
        if slot != cur:
            cuda.syncthreads()
            gbase = cur * 6 * nbins
            for t in range(tid, nbins, nthr):
                if sh[5 * nbins + t] != 0.0:
                    for q in range(6):
                        cuda.atomic.add(out, gbase + q * nbins + t, sh[q * nbins + t])
                for q in range(6):
                    sh[q * nbins + t] = 0.0
            cuda.syncthreads()
            cur = slot
        _accumulate_pair(
            p, pair_i1, pair_i2, pair_ang, pair_shp,
            off1, z1, rc1, dm1, w1, dl1, zq1,
            off2, z2, rc2, dm2, w2, dl2, zq2,
            fparams, iparams, sh, 0, nbins,
        )
    cuda.syncthreads()
    gbase = cur * 6 * nbins
    for t in range(tid, nbins, nthr):
        if sh[5 * nbins + t] != 0.0:
            for q in range(6):
                cuda.atomic.add(out, gbase + q * nbins + t, sh[q * nbins + t])


def _enable_large_shared(kernel, nbytes):
    """Opt a compiled numba kernel in to ``nbytes`` of dynamic shared memory
    (> 48 KiB) through the driver API; numba has no wrapper for this.
    Returns False if anything about it is unavailable (use the global kernel)."""
    try:
        import ctypes

        cufunc = list(kernel.overloads.values())[0]._codelibrary.get_cufunc()
        lib = ctypes.CDLL("libcuda.so.1")
        # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
        return lib.cuFuncSetAttribute(cufunc.handle, 8, ctypes.c_int(nbytes)) == 0
    except Exception:  # pylint: disable=broad-except
        return False


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


class _DeviceXi:
    """Accumulates the per-healpix correlation function on one GPU."""

    PAIRS_PER_BLOCK = int(__import__("os").environ.get("PICCA_GPU_PPB", 32))
    THREADS = 1024

    def __init__(self, flat1, flat2, healpix_list, device=0):
        self.device = device
        self.nbins = cf.num_bins_r_par * cf.num_bins_r_trans
        self.slot_of = {h: s for s, h in enumerate(healpix_list)}
        self.out = cuda.device_array(len(healpix_list) * 6 * self.nbins, np.float64)
        self.out[:] = 0
        self.f1 = flat1.to_device()
        self.f2 = self.f1 if flat2 is flat1 else flat2.to_device()
        fp, ip = _params()
        self.fparams = cuda.to_device(fp)
        self.iparams = cuda.to_device(ip)
        self.shmem = 6 * self.nbins * 8
        self.use_shared = None  # decided by _setup_shared()

    def _setup_shared(self):
        limit = cuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        ok = self.shmem <= limit
        if ok:
            # compile with a no-op launch (npairs = 0), then raise the smem limit
            d = [cuda.to_device(x) for x in (
                np.zeros(1, np.int32), np.zeros(1, np.int32), np.zeros(1),
                np.zeros(1, np.uint8), np.zeros(1, np.int32))]
            _xi_kernel_shared[1, 1, 0, 8](
                *d, 0, 1, *self.f1, *self.f2, self.fparams, self.iparams, 0, self.out
            )
            cuda.synchronize()
            ok = _enable_large_shared(_xi_kernel_shared, self.shmem)
        self.use_shared = ok
        if not ok:
            print("cf_gpu: large shared memory unavailable, using global atomics")

    def add(self, idx1, idx2, ang, shp, slot):
        npairs = idx1.size
        if npairs == 0:
            return
        d = [cuda.to_device(x) for x in (idx1, idx2, ang, shp, slot)]
        ppb = self.PAIRS_PER_BLOCK
        nblocks = (npairs + ppb - 1) // ppb
        args = (
            *d, npairs, ppb, *self.f1, *self.f2,
            self.fparams, self.iparams, self.nbins, self.out,
        )
        if self.use_shared:
            _xi_kernel_shared[nblocks, self.THREADS, 0, self.shmem](*args)
        else:
            _xi_kernel_global[nblocks, BLOCK](*args)

    def host_sums(self):
        cuda.synchronize()
        return self.out.copy_to_host().reshape(-1, 6, self.nbins)


class XiAccumulator:
    """Spreads healpixes over the GPUs of the node and returns the per-healpix results in
    the same form as ``cf.compute_xi``.

    Each GPU is driven by its own thread (the host-to-device copies block the host, so a
    single driving thread keeps only about one GPU busy), and every healpix is dealt out in
    pieces so that the GPUs finish together."""

    QUEUE_DEPTH = 3
    BATCH = 16384  # forest pairs per kernel launch (512 blocks at 32 pairs per block)

    def __init__(self, flat1, flat2, healpix_list, devices=None):
        if devices is None:
            devices = list(range(len(cuda.gpus)))
        self.slot_of = {h: s for s, h in enumerate(healpix_list)}
        self.parts = []
        for dev in devices:
            with cuda.gpus[dev]:
                part = _DeviceXi(flat1, flat2, healpix_list, device=dev)
                # set up (compile, raise the shared-memory limit) here, one device at a time
                part._setup_shared()
                self.parts.append(part)
        self.queues = [queue.Queue(maxsize=self.QUEUE_DEPTH) for _ in self.parts]
        self.errors = []
        self.threads = [
            threading.Thread(target=self._worker, args=(k,), daemon=True)
            for k in range(len(self.parts))
        ]
        for t in self.threads:
            t.start()
        self.next = 0
        self.pending, self.pending_n = [], 0

    def _worker(self, k):
        part = self.parts[k]
        cuda.select_device(part.device)
        try:
            while True:
                item = self.queues[k].get()
                if item is None:
                    break
                part.add(*item)
            cuda.synchronize()
        except BaseException as err:  # pylint: disable=broad-except
            self.errors.append(err)
            while self.queues[k].get() is not None:  # keep draining so add() never blocks
                pass

    def add(self, healpix, idx1, idx2, ang, shp):
        if self.errors:
            raise self.errors[0]
        n = idx1.size
        if n == 0:
            return
        slot = np.full(n, self.slot_of[healpix], dtype=np.int32)
        self.pending.append((idx1, idx2, ang, shp, slot))
        self.pending_n += n
        if self.pending_n >= self.BATCH:
            self._send()

    def _send(self):
        """Hand the pending healpixes to the next GPU as one launch (enough blocks to fill it)."""
        if not self.pending:
            return
        batch = [np.concatenate(cols) for cols in zip(*self.pending)]
        self.pending, self.pending_n = [], 0
        self.queues[self.next % len(self.parts)].put(tuple(batch))
        self.next += 1

    def result(self, healpix_list):
        """Same structure as ``np.array([cf.compute_xi(h) for h in healpixs])``."""
        self._send()
        for q in self.queues:
            q.put(None)
        for t in self.threads:
            t.join()
        if self.errors:
            raise self.errors[0]
        host = None
        for part in self.parts:
            with cuda.gpus[part.device]:
                h = part.host_sums()
            host = h if host is None else host + h
        res = []
        for h in healpix_list:
            xx, ww, pp, tt, zz, nn = (a.copy() for a in host[self.slot_of[h]])
            m = ww > 0
            xx[m] /= ww[m]
            pp[m] /= ww[m]
            tt[m] /= ww[m]
            zz[m] /= ww[m]
            res.append((ww, xx, pp, tt, zz, np.rint(nn).astype(np.int64)))
        return res

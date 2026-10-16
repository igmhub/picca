# GPU support for the picca correlation scripts (branch `gpu-port`)

This branch adds an opt-in `--gpu` option to the correlation scripts, so the pair loops run on the GPUs of a
Perlmutter GPU node instead of on the CPU workers, and an opt-in `--seed-per-healpix` option for the distortion
matrices. Without these options nothing changes: same code path, same results, same output format.

## What is new

| Script | New option(s) | What runs on the GPU |
|---|---|---|
| `picca_cf.py` | `--gpu` | forest x forest pixel pairs, histogrammed into the correlation bins |
| `picca_xcf.py` | `--gpu` | forest x quasar pixel pairs |
| `picca_dmat.py` | `--gpu`, `--seed-per-healpix` | distortion matrix |
| `picca_xdmat.py` | `--gpu`, `--seed-per-healpix` | cross-correlation distortion matrix |
| `picca_metal_dmat.py` | `--gpu`, `--seed-per-healpix` | metal distortion matrix |
| `picca_metal_xdmat.py` | `--seed-per-healpix` | (not ported to the GPU) |

New modules: `py/picca/cf_gpu.py`, `py/picca/xcf_gpu.py`, `py/picca/dmat_gpu.py`. The only edits to existing
modules (`cf.py`, `xcf.py`) are the `seed_per_healpix` hooks (off by default). The edits to the `bin/` scripts are
the new options. `py/picca/tests/test_4_gpu.py` holds the new tests.

Not ported, deliberately left on the CPU: `wick`, `cf1d`, `picca_co`, `cf_angl`, `fast_metal`. The angular
correlation (`cf.ang_correlation`) raises with `--gpu`.

## Requirements and use

- A CUDA GPU and a numba with CUDA support (`numba.cuda`). Nothing else is needed (no cupy). It works in the
  `vega_master` environment of the Lya group and in the DESI 26.9 environment.
- All GPUs of the node are used (healpixes are dealt out round-robin).
- Use it by adding `--gpu`, for example

      picca_cf.py --gpu --out cf_lya_lya.fits.gz --in-dir deltas_lya/Delta --rp-min 0 --rp-max 200 --rt-max 200 \
        --nt 50 --np 50 --fid-Om 0.315191867993297 --fid-Or 7.96641826260616e-05 --nproc 128 ...

  `--nproc` still sets the number of CPU workers, which find the neighbouring forests and (for the distortion
  matrices) draw the pair rejection; the GPU runs the pair kernel.
- lyatools: a `gpu = True` setting in `[picca_corr]` adds `--gpu` to the generated scripts (see the lyatools branch).

## How it works

- The forests are flattened once into arrays. CPU workers (forked **before** any CUDA call) find the neighbours of
  each healpix with array operations and return only index arrays; the main process launches the kernels.
- `cf`/`xcf`: each block takes 32 forest pairs and accumulates the six histogram planes (xi, weight, r_par,
  r_trans, z, count) in about 120 KB of shared memory, then adds them to the global arrays with one atomic per
  bin. (`xcf` treats each quasar as a one-pixel forest with delta 1, and reuses the `cf` kernel.)
- `dmat`/`xdmat`/`metal_dmat`: one block per forest pair; see the docstring of `dmat_gpu.py`.
- Floating point sums are in a different order from the CPU code (atomic adds), so the results agree to about
  1e-13 relative, not bit for bit. Pair counts (`NB`) are identical.

## `--seed-per-healpix`

The CPU distortion matrices seed the random pair rejection once per chunk of healpixes with the first healpix of
the chunk (`np.random.seed(healpixs[0])`), so the result depends on `--nproc`. With `--seed-per-healpix` the
generator is seeded with each healpix number, and the result is independent of `--nproc` and of CPU or GPU.
Default off, so existing results are unchanged.

## What was checked

- `py/picca/tests/test_4_gpu.py` (16 tests, skipped without a CUDA device): GPU against CPU for `cf`, `xcf`, `dmat`,
  `xdmat` and `metal_dmat` on small synthetic data, relative tolerance 1e-10, pair counts exact. Run with
  `python -m pytest py/picca/tests/test_4_gpu.py`. The existing tests (`test_2_pk1d.py`, `test_3_cor.py`,
  `test_civ_blinding.py`) pass on the merged branch (151 tests with the new ones).
- Full-size jobs of mock 0 of the Y3 lyacolore mocks (production deltas): `cf`, `xcf`, `dmat`, `xdmat` and
  `metal_dmat` agree with the production outputs to about 1e-13 relative (`compare_fits`-style comparison of every
  column), in the DESI 26.9 environment and in `vega_master`.
- A whole-pipeline test on a 1024^3 mock (CoLoRe, LyaCoLoRe, quickquasars and picca run once with the original
  code and once with these changes, on different inputs that differ at the 1e-6 level): the correlations agree
  within 1e-6 relative with identical pair counts, the distortion matrices within 1e-8.

## Timings

Production inputs, mock 0, one GPU node (4 x A100): `dmat_lya_lya` 4 h 10 min on the CPU (production job) to
12 min; `xcf_lya_qso` 6.0 to 4.2 min, `xcf_lyb_qso` 2.4 to 1.7 min.

1024^3 pipeline test (one node per run, A100-40GB node against a CPU node running the same script):

| Step | CPU | GPU |
|---|---|---|
| `cf_lya_lya` | 1553 s | 366 s |
| `xcf_lya_qso` | 357 s | 316 s |
| `xdmat_lya_qso` | 2081 s | 354 s |
| `dmat_lya_lya` | 4 h 10 min (production record, not rerun) | 747 s |

`xcf` gains little because most of its time is the neighbour search on the CPU workers while the GPU waits. All
the scripts still read the deltas serially at the start (about 2 to 3 min per job); that is not changed here.
Single runs; the hardware is not identical (a CPU node and a GPU node).

## Commits (on top of picca 9.21.3, merged with `main` at 9.21.4)

`ce1311b0` cf pair kernel; `5cf262a4` shared-memory histogram; `bb1a2bf1` round-robin over GPUs; `c74eab8e`,
`97dc0e5d`, `d69adcd4`, `26bb34b8` dmat and metal_dmat kernels; `ce35aeb1` xdmat; `0d386e99`, `fcc5f724` cf
launch batching and neighbour search; `58cb5551` xcf; `ffeaf243` `--seed-per-healpix`; `010dcd91` tests.

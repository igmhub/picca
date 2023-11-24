# picca
[![Coverage Status](https://coveralls.io/repos/github/igmhub/picca/badge.svg?branch=master)](https://coveralls.io/github/igmhub/picca?branch=master)

Package for Igm Cosmological-Correlations Analyses.

This package contains tools used for the analysis of the Lyman-alpha forest sample from the extended Baryon Oscillation Spectroscopic Survey (eBOSS) and the Dark Energy Spectroscopic Instrument (DESI). Here you will find tools to

- fit continua of forests
- compute correlation functions (1D and 3D) and power-spectra (1D)
- compute covariance matrices
- fit models for the correlation functions

The current reference is du Mas des Bourboux et al. 2020 (https://arxiv.org/abs/2007.08995).

## Installation
First, create a clean environment with `version`>=3.9:
```
conda create -n my_picca_env python==version
conda activate my_picca_env
```
If you already have an environment, you just need to activate it.
After you have the environment, you can install picca with:
```
pip install picca
```
If you are a developer, or want the most recent version of picca, you can download and install manually:
```
git clone https://github.com/igmhub/picca.git
cd picca
pip install -e .
```
Optionally, you can add the path to picca to your bashrc:
```
export PICCA_BASE=<path to your picca>
```
Or you can add `picca/py/` to your `PYTHONPATH`. Both of these are optional and picca will work without them.

If you are at working at NERSC, we recommend to keep everything clean by adding a function like this in your bashrc:
```
picca_env () {
    module load python
    conda activate my_picca_env
}
```
Whenever you need picca just write:
```
picca_env
```
This is cleaner than directly adding the commands to the bashrc file, and avoids potential issues with the transition to Perlmutter.

If you want to compute models for the correlations computed with picca, or you want to fit these correlations, see https://github.com/andreicuceu/vega.

If you are running MPI code (only needed for some tasks in fitter2), see https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment. If want to run the PolyChord sampler for fitter2, see https://github.com/andreicuceu/fitter2_tutorial. Note that fitter2 is deprecated, and will be removed in the future.

If you need to run the "picca_compute_pk_pksb.py" script you will also need to install the following packages:
```
pip install camb
pip install cython
pip install nbodykit
```

## Reproducing the BAO measurement in eBOSS DR16 (du Mas des Bourboux et al. 2020)

Picca v4.0 was used in du Mas des Bourboux et al. (2020) to compute the final Lyman-alpha BAO measurement from eBOSS DR16.

You can find a tutorial describing the different steps needed to reproduce the analysis (starting from the public catalogs) in `tutorials/eboss_dr16`.


## Examples

example run over 1000 spectra (the DLA catalog is not required):

### delta field
```
picca_delta_extraction.py config.ini
```

* To reproduce the eBOSS analysis (du Mas des Bourboux et al. 2020) this needs to be run four times, two for calibration purposes, one for the Lyman $\alpha$ region and one for the Lyman $\beta$ region

* Check the tutorial `picca_delta_extraction_configuration_tutorial` to review the available options. Find it under `tutorials/delta_extraction`

* Check the folder `examples/delta_extraction` with examples to reproduce the eBOSS analysis

### old delta field (deprecated)

```
picca_deltas.py
--in-dir data/
--drq ../DR14Q_v1_1.fits
--dla-vac ../dlas/DLA_DR14_v1b.dat
--out-dir deltas/
--mode pix
```

*   for eBOSS, currently `--mode` can be  `spplate`, `spec`, `pix`, or `spcframe`, all but the first 2 are about to be retired
*   NOTE: reading the spec files is *very* slow
*   for DESI currently `--mode` can be `desi_mocks` (for reading healpix based mocks), `desi_survey_tilebased` (for reading cumulative tiles directories and coadding data across tiles) or `desi_sv_no_coadd` (for reading tile based directories without coadding, will probably be retired)
*   `--in-dir` points to the directory containing the data (in case of `desi_survey_tilebased` the full path until and including `cumulative` is needed, in case of `desi-mocks` the full path until `spectra-16`)
*   the `--drq` points towards a quasar catalog in either the DESI format or eBOSS format

### correlation function

```bash
picca_cf.py
--in-dir deltas/
--out cf.fits.gz
--nside 32
```
*   `nside` determines the healpixelization used for the subsamples. `nside=32` gives ~3200 subsamples for DR12.

### distortion matrix

```bash
picca_dmat.py
--in-dir deltas/
--out dmat.fits.gz
--rej 0.95
```

*   `--rej` is 1-fraction of pairs used for the calculation

### wick covariance (optional)

Only T123 implemented

```bash
# first calculate cf_1d from data
picca_cf1d.py
--in-dir deltas/
--out cf1d.fits.gz

# then use it for wick
picca_wick.py
--in-dir deltas/
--out t123.fits.gz
--rej 0.999
--cf1d cf1d.fits.gz

## use the export script to export to picca fitter format
picca_export.py
--data cf.fits.gz
--dmat dmat.fits.gz
--out cf-exp.out.gz
```

### Name of tags

The tags name follow the names of the king of France:<br/>
https://fr.wikipedia.org/wiki/Liste_des_monarques_de_France#Liste_des_monarques_de_France

### For Developers
Before submitting a PR please make sure to:
1. Check the tutorials. Update them if necessary (typically the tutorial `picca_delta_extraction_configuration_tutorial` will need to be updated.
2. Update the data model
3. For every file you have modified run
   ```
   yapf --style google file.py -i
   ```
   to ensure the coding styles are maintained.
4. Consider using pylint to help in the debug process. From the repo folder run
   ```
   pylint py/picca/delta_extraction/
   pylint py/picca/pk1d/
   ```
   depending on the module you are working on.
   
When merging PRs (or committing to master directly):
- by default the patch version is increased via a github action, so every change of master will generate a new version
This behaviour can be changed by adding one of the following to the commit-msg of the merge commit:
- by specifying [bump minor] or [bump major] a new minor or major version will be generated instead, but tags and releases need to be created manually (and are auto-pushed to pypi when they are created)
- by specifying [no bump] the version bump can be circumvented altogether when some other behaviour is wanted, in that case bump2version should be run manually

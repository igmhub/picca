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

download
```bash
git clone https://github.com/igmhub/picca.git
```

add to your bashrc
```bash
export PICCA_BASE=<path to your picca>
```

then make sure you have all required modules by running
```bash
pip install -r requirements.txt --user
```

and finally run
```bash
python setup.py install --user
```
(assuming you run as user; for a system-wide install omit `--user` option).

Alternatively, you can just add `picca/py/` to your `PYTHONPATH`.

## Examples

example run over 1000 spectra (the DLA catalog is not required):

### delta field

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

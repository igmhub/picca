
# picca

Package for Igm Cosmological-Correlations Analyses


requirements:
* python 2.7
* scipy 0.17.0 or later
* iminuit 1.2 or later
* fitsio
* healpy
* numba
* multiprocessing
* configargparse
* h5py

## Installation

download
```
git clone https://github.com/igmhub/picca.git
```

add to your bashrc
```
export PICCA_BASE=<path to your picca>
```

then make sure you have all required modules by running
```
pip install -r requirements.txt --user
```

and finally run
```
python setup.py install --user
```
(assuming you run as user; for a system-wide install omit `--user` option).

Alternatively, you can just add `picca/py/` to your `PYTHONPATH`.

## Examples

example run over 1000 spectra (the DLA catalog is not required):

### delta field

```
python bin/do_deltas.py --in-dir data/ --drq ../DR14Q_v1_1.fits --dla-vac ../dlas/DLA_DR14_v1b.dat --out-dir deltas/ --mode pix
```

* --mode can be pix (Anze/Jose format), spec (spec- files) or corrected-spec (corrected-spec files)
* --in-dir points to the directory containing the data
* NOTE: reading the spec files is *very* slow

### correlation function

```
python bin/do_cf.py --in-dir deltas/ --out cf.fits.gz --nside 32
```
* nside determines the healpixelization used for the subsamples. nside=32 gives ~3200 subsamples for DR12.

### distortion matrix

```
python bin/do_dmat.py --in-dir deltas/ --out dmat.fits.gz --rej 0.95
```

* `--rej` is 1-fraction of pairs used for the calculation

### wick covariance (optional).

Only T123 implemented

```
# first calculate cf_1d from data
python bin/do_cf1d.py --in-dir deltas/ --out cf1d.fits.gz

# then use it for wick
python bin/do_wick.py --in-dir deltas/ --out t123.fits.gz --rej 0.999 --cf1d cf1d.fits.gz


## use the export script to export to picca fitter format
python bin/export --data cf.fits.gz --dmat dmat.fits.gz --out cf-exp.out.gz
```

### Name of tags

The tags name follow the names of the king of France:
https://fr.wikipedia.org/wiki/Liste_des_monarques_de_France#Liste_des_monarques_de_France

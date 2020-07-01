Instructions to reproduce the DR12 auto-correlation (Bautista et al. 2017).

Andreu Font-Ribera was able to follow these instructions at NERSC in July 2020.


--- Setup ---

Setup an environmental variable pointing to your work directory where you want
to reproduce the results, as in:
export REDO_B17='/project/projectdirs/eboss/lya_forest/redo_dr12/redo_B17/'


--- Create output folders ---

Go to the analysis folder
cd $REDO_B17

Create a bunch of folders that will be used to store the results:
mkdir Delta_LYA Delta_LYA/Delta Delta_LYA/Log/
mkdir Delta_calibration Delta_calibration/Delta Delta_calibration/Log/
mkdir Delta_calibration2 Delta_calibration2/Delta Delta_calibration2/Log/
mkdir Correlations Fits


--- Generating the delta files ---

If running at NERSC, you probably want to start an interactive session:
salloc -N 1 -C haswell -q interactive -t 04:00:00

Run picca_deltas using different scripts
Note that you will have these in serie, you have to wait for each to finish
before starting the next run 

picca_deltas.py `more $PICCA_DR12/Scripts_B17/produce_delta_calib.txt` > info_calib &
picca_deltas.py `more $PICCA_DR12/Scripts_B17/produce_delta_calib2.txt` > info_calib2 &
picca_deltas.py `more $PICCA_DR12/Scripts_B17/produce_delta_LYA.txt` > info_LYA &


--- Inspecting the delta files ---

Make the usual plot inspecting the result of continuum fitting:
python $PICCA_DR12/Scripts_B17/look_delta_attributes.py


--- Computing the Lya auto-correlation ---

Start another interactive session at NERSC, for instance running:
salloc -N 1 -C haswell -q interactive -t 04:00:00

Run a script to measure the correlation, distortion and metal matrix:
python -u $PICCA_DR12/Scripts_B17/send_cor.py > info_cf &  


--- Inspecting the measured correlations ---

Install baoutil from Julien's branch: 
https://github.com/julienguy/baoutil/tree/julien

plot_xi.py -d Correlations/cf.fits.gz --mu "0.95:1.0,0.8:0.95,0.5:0.8,0.0:0.5"


--- Run BAO fits ---

picca_fitter2.py Fits/chi2_cf_v0.ini
picca_fitter2.py Fits/chi2_xcf_v0.ini


--- Inspect results from BAO fits ---

extract_fit_pars.py -i Fits/results_cf_v3_rej099_lyaonly.h5
extract_fit_pars.py -i Fits/results_xcf_v3_rej098_lyaonly.h5


--- You can extract to ASCII the results as well ---

extract_model.py -i Fits/results_cf_v3_rej099_lyaonly.h5 -o Fits/results_cf_v3_rej099_lyaonly.txt
extract_model.py -i Fits/results_xcf_v3_rej098_lyaonly.h5 -o Fits/results_xcf_v3_rej098_lyaonly.txt


--- You can overplot data and model like this ---

plot_xi.py -d Correlations/e_cf_v3_rej099.fits.gz --mu "0.95:1.0,0.8:0.95,0.5:0.8,0.0:0.5" --model Fits/results_cf_v3_rej099_lyaonly.txt 
plot_xi.py -d Correlations/e_xcf_v3_rej098.fits.gz --mu "0.96:1.0,0.8:0.96,0.5:0.8,0.0:0.5" --abs --rrange "0:200" --model Fits/results_xcf_v3_rej098_lyaonly.txt



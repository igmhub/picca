Instructions to reproduce the DR12 cross-correlation (du Mas des Bourboux et al. 2017).

Andreu Font-Ribera was able to follow these instructions at NERSC in July 2020.


--- Setup ---

Setup an environmental variable pointing to your work directory where you want
to reproduce the results, as in:
export REDO_dMdB17='/project/projectdirs/eboss/lya_forest/redo_dr12/redo_dMdB17/'


--- Create output folders ---

Go to the analysis folder
cd $REDO_dMdB17

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

picca_deltas.py `more $PICCA_DR12/Scripts_dMdB17/produce_delta_calib.txt` > info_calib &
picca_deltas.py `more $PICCA_DR12/Scripts_dMdB17/produce_delta_calib2.txt` > info_calib2 &
picca_deltas.py `more $PICCA_DR12/Scripts_dMdB17/produce_delta_LYA.txt` > info_LYA &


--- Inspecting the delta files ---

Make the usual plot inspecting the result of continuum fitting:
python $PICCA_DR12/Scripts_dMdB17/look_delta_attributes.py


--- Computing the Lya auto-correlation ---

Start another interactive session at NERSC, for instance running:
salloc -N 1 -C haswell -q interactive -t 04:00:00

Run a script to measure the correlation, distortion and metal matrix:
python -u $PICCA_DR12/Scripts_dMdB17/send_cor.py > info_xcf &  

Note that there will be a lot of numba deprecation warnings.


--- Inspecting the measured correlations ---

Install baoutil from Julien's branch: 
https://github.com/julienguy/baoutil/tree/julien

plot_xi.py -d $REDO_dMdB17/Correlations/xcf-exp.fits.gz --mu "0.95:1.0,0.8:0.95,0.5:0.8,0.0:0.5" --abs --rrange "0:200" 


--- Run BAO fits ---

picca_fitter2.py $PICCA_DR12/Scripts_dMdB17/chi2_xcf_baseline.ini > info_xcf_baseline &


--- Inspect results from BAO fits ---

extract_fit_pars.py -i $REDO_dMdB17/Fits/results_xcf_baseline.h5


--- You can extract to ASCII the results as well ---

extract_model.py -i $REDO_dMdB17/Fits/results_xcf_baseline.h5 -o Fits/results_xcf_baseline.txt


--- You can overplot data and model like this ---

plot_xi.py -d $REDO_dMdB17/Correlations/xcf-exp.fits.gz --mu "0.95:1.0,0.8:0.95,0.5:0.8,0.0:0.5" --model $REDO_dMdB17/Fits/results_xcf_baseline.txt --abs --rrange "0:200" --out-figure $REDO_dMdB17/Correlations/xcf


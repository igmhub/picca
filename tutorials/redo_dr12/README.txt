These are the instructions to reproduce the DR12 results using Picca.

Based on the work by Andreu Font-Ribera at the end of 2019, to add a comparison
between DR12 and DR16 in the final eBOSS analysis paper.

The results are not identic to those published in the DR12 analyses because
of several reasons, described in more detail in the separate README files:
 - README_B17 for the auto-correlation (Bautista et al. 2017)
 - README_dMdB17 for the cross-correlation (du Mas des Bourboux et al. 2017)


--- Setup environmental variables ---

For both analyses you need to setup the following environmental variables:

Setup an environmental variable to point to the DR12 spectra (DR13 pipeline):
export EBOSS_DR13='/global/project/projectdirs/cosmo/data/sdss/dr13/eboss/'
export EBOSS_v5_9_0=$EBOSS_DR13'/spectro/redux/v5_9_0/'


--- Setup Picca ---

Of course, you need to have Picca installed. If you do not have it,
follow the instructions here:  https://github.com/igmhub/picca
In order to reproduce the DR12 results, it is important to use the redo_dr12
branch of Picca software, released in GitHub as _____.

You will also need to setup several environmental variables for Picca:
export PICCA_BASE=<path to your picca>
export PYTHONPATH=$PICCA_BASE/py:$PYTHONPATH
export PATH=$PICCA_BASE/bin:$PATH
export PICCA_DR12=$PICCA_BASE'/tutorials/redo_dr12/'

To make some of the plots, you will also need to install baoutil from 
Julien's branch:
https://github.com/julienguy/baoutil/tree/julien


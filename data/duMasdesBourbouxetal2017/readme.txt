
Gathering of results of H. du Mas des Bourboux et al. 2017
https://arxiv.org/abs/

BOSS_DR12/correlations/
	The calculated auto and cross correlation for
	r_paral \in [-200,200] h^{-1}~Mpc and for r_perp \in [0,200] h^{-1}~Mpc
	with bin size 4 h^{-1}~Mpc

BOSS_DR12/model/
	CAMB power spectrum at z = 2.40
	generated with Planck 2015

fits/
	standard fit to the auto, cross and combined of the two
	In each folder:
		- *.ini gives the config file to use in the fitter
		- *.save.pars gives the best fit parameters
		- *.save.pars.cor gives the correlation matrix of the best fit parameters
		- *.combined_fit.chisq gives the total chi2 at best fit
		- *..ap.scan.dat gives the 1D scan of chi2 allong the alpha_parallel parameter
		- *..at.scan.dat gives the 1D scan of chi2 allong the alpha_perp parameter
		- *..at.ap.scan.dat gives the 2D scan of chi2 allong the (alpha_perp,alpha_paralle) parameters


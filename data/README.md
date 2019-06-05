# Published data

## Tutorial

In each folder:

*   `*.chisq` gives the total chi2 at best fit

*   `*.at.ap.scan.dat` gives the 2D scan of chi2 allong the
        `(alpha_parallel,alpha_perp)` parameters

*   `*.save.pars` gives the best fit results for alpha_parallel
    and alpha_perp

*   `*.fiducial` gives the fiducial cosmology D_H/r_d and D_M/r_d

To plot the chi2-scan, you can run for example:
```bash
tutorials/data/picca_plot_chi2_surface.py
--chi2scan data/deSainteAgatheetal2019/auto_alone_stdFit/auto_alone_stdFit.ap.at.scan.dat
data/Blomqvistetal2019/cross_alone_stdFit/cross_alone_stdFit.ap.at.scan.dat
data/deSainteAgatheetal2019/combined_stdFit/combined_stdFit.ap.at.scan.dat
--label auto-DR14 cross-DR14 combined-DR14
```

![DR14-chi2scan](/tutorials/data/DR14-chi2-scan-ap-at.png)

## DR14

### Results of V. de Sainte Agathe et al. 2019
arXiv:1904.03400<br/>
In `deSainteAgatheetal2019/`:

*   "auto\_alone\_stdFit": fit results of the combined Lya absorption in Lya region
    auto-correlation (Lya(lya)xLya(Lya)) and  Lya absorption in Lya
    region and Lya absorption in Lyb region correlation function
    (Lya(Lya)xLya(Lyb))

*   "combined\_stdFit": combined with the cross-correlation function from Blomqvist et al. 2019

### Results of M. Blomqvist et al. 2019
arXiv:1904.03430<br/>
In `Blomqvistetal2019/`:

*   "cross\_alone\_stdFit": standard fit results to quasar-(Lya+Lyb) regions cross-correlation
    data

*   "combined\_stdFit": combined with the correlations functions from de Sainte Agathe et al. 2019.

## DR12

### Results of J.E. Bautista et al. 2017
arXiv:1702.00176<br/>
In `Bautistaetal2017/fits/`:

*   "physical": fit results to lyman-alpha forest auto-correlation data

### Results of H. du Mas des Bourboux et al. 2017
arXiv:1708.02225<br/>
In `duMasdesBourbouxetal2017/fits/`:

*   "cross\_alone\_stdFit": standard fit results to quasar-lyman-alpha forest cross-correlation data

*   "combined\_stdFit": combined with lyman-alpha forest auto-correlation

# Published data

## Tutorial

In each folder:

*   `*.chisq` gives the total chi2 at best fit

*   `*.at.ap.scan.dat` gives the 2D scan of chi2 allong the
        `(alpha_parallel,alpha_perp)` parameters

*   `*.save.pars` gives the best fit results for alpha\_parallel
    and alpha_perp

*   `*.fiducial` gives the fiducial cosmology D\_H/r\_d and D\_M/r\_d

To plot the chi2-scan, you can run for example:
```bash
tutorials/data/picca_plot_chi2_surface.py --chi2scan \
data/duMasdesBourbouxetal2020/auto_full_stdFit/autofull.ap.at.scan.dat \
data/duMasdesBourbouxetal2020/cross_full_stdFit/crossfull.ap.at.scan.dat \
data/duMasdesBourbouxetal2020/combined_stdFit/combined.ap.at.scan.dat \
--label auto-DR16 cross-DR16 combined-DR16
```

![DR16-chi2scan](/tutorials/data/DR16-chi2-scan-ap-at.png)

## DR16

### Results of H. du Mas des Bourboux et al. 2020 ([arXiv:2007.08995](https://arxiv.org/abs/2007.08995), published in ApJ)
In ![duMasdesBourbouxetal2020](/data/duMasdesBourbouxetal2020/):

*   ![auto\_full\_stdFit](/data/duMasdesBourbouxetal2020/auto_full_stdFit/): Result of the full auto-correlation with a combined fit of:
    * auto-correlation of Lya absorption in the Lya region (Lya(Lya)xLya(Lya))
    * auto-correlation of Lya absorption in the Lya region with Lya absorption in the Lyb region (Lya(Lya)xLya(Lya))

*   ![cross\_full\_stdFit](/data/duMasdesBourbouxetal2020/cross_full_stdFit/): Result of the full cross-correlation with a combined fit of:
    * cross-correlation of Lya absorption in the Lya region with quasars (Lya(Lya)xQSO)
    * cross-correlation of Lya absorption in the Lyb region with quasars (Lya(Lyb)xQSO)

*   ![combined\_stdFit](/data/duMasdesBourbouxetal2020/combined_stdFit/): Result of the fit to the four previously defined
    different correlations

## DR14

### Results of V. de Sainte Agathe et al. 2019 ([arXiv:1904.03400](https://arxiv.org/abs/1904.03400), published in A&A)
In ![deSainteAgatheetal2019](/data/deSainteAgatheetal2019/):

*   ![auto\_alone\_stdFit](/data/deSainteAgatheetal2019/auto_alone_stdFit/): fit results of the combined Lya absorption in Lya region
    auto-correlation (Lya(lya)xLya(Lya)) and  Lya absorption in Lya
    region and Lya absorption in Lyb region correlation function
    (Lya(Lya)xLya(Lyb))

*   ![combined\_stdFit](/data/deSainteAgatheetal2019/combined_stdFit/): combined with the cross-correlation function from Blomqvist et al. 2019

### Results of M. Blomqvist et al. 2019 ([arXiv:1904.03430](https://arxiv.org/abs/1904.03430), published in A&A)
In ![Blomqvistetal2019/](/data/Blomqvistetal2019/):

*   ![cross\_alone\_stdFit](/data/Blomqvistetal2019/cross_alone_stdFit): standard fit results to quasar-(Lya+Lyb) regions cross-correlation
    data

*   ![combined\_stdFit](/data/Blomqvistetal2019/combined_stdFit/): combined with the correlations functions from de Sainte Agathe et al. 2019.

## DR12

### Results of J.E. Bautista et al. 2017 ([arXiv:1702.00176](https://arxiv.org/abs/1702.00176), published in A&A)
In ![Bautistaetal2017/fits](/data/Bautistaetal2017/fits/):

*   ![physical](/data/Bautistaetal2017/fits/physical/): fit results to lyman-alpha forest auto-correlation data

### Results of H. du Mas des Bourboux et al. 2017 ([arXiv:1708.02225](https://arxiv.org/abs/1708.02225), published in A&A)
In ![duMasdesBourbouxetal2017/fits](/data/duMasdesBourbouxetal2017/fits/):

*   ![cross\_alone\_stdFit](/data/duMasdesBourbouxetal2017/fits/cross_alone_stdFit/): standard fit results to quasar-lyman-alpha forest cross-correlation data

*   ![combined\_stdFit](/data/duMasdesBourbouxetal2017/fits/combined_stdFit/): combined with lyman-alpha forest auto-correlation

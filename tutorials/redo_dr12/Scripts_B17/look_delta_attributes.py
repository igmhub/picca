import fitsio
import matplotlib.pyplot as plt
import os


def plot_result(basedir,saveplots=False):

    path = basedir+'/Log/delta_attributes.fits.gz'
    data = fitsio.FITS(path)

    ### Stack
    loglam = data[1]['LOGLAM'][:]
    stack  = data[1]['STACK'][:]
    cut = (stack!=0.) & (data[1]['WEIGHT'][:]>0.)
    loglam = loglam[cut]
    stack  = stack[cut]
    plt.plot(10.**loglam, stack, linewidth=4,marker='o')
    plt.grid()
    plt.xlabel(r'$\lambda_{\mathrm{Obs.}} \, [\AA]$',fontsize=14)
    plt.ylabel(r'$\mathrm{\overline{Flux}}$',fontsize=14)
    if saveplots:
        plt.savefig(basedir+'/Log/mean_flux.png')
    plt.show()

    ### ETA
    loglam    = data[2]['LOGLAM'][:]
    eta       = data[2]['ETA'][:]
    nb_pixels = data[2]['NB_PIXELS'][:]
    cut = (nb_pixels>0.)&(eta!=1.)
    loglam = loglam[cut]
    eta    = eta[cut]
    plt.errorbar(10.**loglam, eta, linewidth=4)
    plt.grid()
    plt.xlabel(r'$\lambda_{\mathrm{Obs.}} \, [\AA]$',fontsize=14)
    plt.ylabel(r'$\eta$',fontsize=14)
    if saveplots:
        plt.savefig(basedir+'/Log/eta.png')
    plt.show()

    ### VAR_LSS
    loglam    = data[2]['LOGLAM'][:]
    var_lss   = data[2]['VAR_LSS'][:]
    nb_pixels = data[2]['NB_PIXELS'][:]
    cut       = (nb_pixels>0.)&(var_lss!=0.1)
    loglam    = loglam[cut]
    var_lss   = var_lss[cut]
    plt.errorbar(10.**loglam, var_lss, linewidth=4)
    plt.grid()
    plt.xlabel(r'$\lambda_{\mathrm{Obs.}} \, [\AA]$',fontsize=14)
    plt.ylabel(r'$\sigma^{2}_{\mathrm{LSS}}$',fontsize=14)
    if saveplots:
        plt.savefig(basedir+'/Log/var_lss.png')
    plt.show()

    ### FUDGE
    loglam    = data[2]['LOGLAM'][:]
    fudge     = data[2]['FUDGE'][:]
    nb_pixels = data[2]['NB_PIXELS'][:]
    cut       = (nb_pixels>0.)&(fudge!=1.e-7)
    loglam    = loglam[cut]
    fudge     = fudge[cut]
    plt.errorbar(10.**loglam, fudge, linewidth=4)
    plt.grid()
    plt.xlabel(r'$\lambda_{\mathrm{Obs.}} \, [\AA]$',fontsize=14)
    plt.ylabel(r'$\mathrm{Fudge}$',fontsize=14)
    if saveplots:
        plt.savefig(basedir+'/Log/fudge.png')
    plt.show()

    ### Mean cont
    loglam_rest = data[3]['LOGLAM_REST'][:]
    mean_cont   = data[3]['MEAN_CONT'][:]
    cut = (mean_cont!=0.) & (data[3]['WEIGHT'][:]>0.)
    loglam_rest = loglam_rest[cut]
    mean_cont   = mean_cont[cut]
    plt.plot(10.**loglam_rest, mean_cont, linewidth=4,marker='o')
    plt.grid()
    plt.xlabel(r'$\lambda_{\mathrm{R.F.}} \, [\AA]$', fontsize=14)
    plt.ylabel(r'$\mathrm{\overline{Flux}}$', fontsize=14)
    if saveplots:
        plt.savefig(basedir+'/Log/mean_cont.png')
    plt.show()

    return

basedir=os.environ['REDO_B17']
plot_result(basedir=basedir+'Delta_LYA/',saveplots=True)


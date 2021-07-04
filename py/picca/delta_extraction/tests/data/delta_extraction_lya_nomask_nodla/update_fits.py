import os
import astropy.io.fits as fits
import numpy as np


filenames = ["delta-28.fits.gz", "delta-45.fits.gz"]

for filename in filenames:
    hdul = fits.open(filename)

    for hdu in hdul[1:]:

        hdu.header["TUNIT4"] = 'Flux units'
        hdu.header["LOS_ID"] = hdu.header["THING_ID"]

        if "PMF"in hdu.header:
            pmf = hdu.header["PMF"]
            hdu.header["PLATE"] = f"{int(pmf.split('-')[0]):04d}"
            hdu.header["MJD"] = f"{int(pmf.split('-')[1]):05d}"
            hdu.header["FIBERID"] = f"{int(pmf.split('-')[2]):04d}"
            del hdu.header["PMF"]

        if "ORDER" in hdu.header:
            del hdu.header["ORDER"]

        if hdu.header["THING_ID"] in [428690499, 429522561, 434205493]:
            hdu.header["PLATE"] = "3655-3657"
            hdu.header["MJD"] = "55240-55244"
            if hdu.header["THING_ID"] == 428690499:
                hdu.header["FIBERID"] = "0054-0660"
            elif hdu.header["THING_ID"] == 429522561:
                hdu.header["FIBERID"] = "0179-0540"
            elif hdu.header["THING_ID"] == 434205493:
                hdu.header["FIBERID"] = "0890-0552"
        elif hdu.header["THING_ID"] in [428035498, 430151253, 430805392,
                                        431936632, 427513857, 427878017,
                                        428707431, 429231814, 429522957]:
            hdu.header["PLATE"] = "3655-9367"
            hdu.header["MJD"] = "55240-57758"
            if hdu.header["THING_ID"] == 428035498:
                hdu.header["FIBERID"] = "0344-0482"
            elif hdu.header["THING_ID"] == 430151253:
                hdu.header["FIBERID"] = "0370-0500"
            elif hdu.header["THING_ID"] == 430805392:
                hdu.header["FIBERID"] = "0273-0601"
            elif hdu.header["THING_ID"] == 431936632:
                hdu.header["FIBERID"] = "0368-0538"
            elif hdu.header["THING_ID"] == 427513857:
                hdu.header["FIBERID"] = "0310-0457"
            elif hdu.header["THING_ID"] == 427878017:
                hdu.header["FIBERID"] = "0228-0446"
            elif hdu.header["THING_ID"] == 428707431:
                hdu.header["FIBERID"] = "0186-0386"
            elif hdu.header["THING_ID"] == 429231814:
                hdu.header["FIBERID"] = "0100-0345"
            elif hdu.header["THING_ID"] == 429522957:
                hdu.header["FIBERID"] = "0198-0396"
        elif hdu.header["THING_ID"] in [434932660]:
            hdu.header["PLATE"] = "3655-9367-10656"
            hdu.header["MJD"] = "55240-57758-58163"
            if hdu.header["THING_ID"] == 434932660:
                hdu.header["FIBERID"] = "0990-0836-0366"

    names = sorted([hdul[i].header["EXTNAME"] for i in range(1, len(hdul))])
    new_hdul = fits.HDUList([hdul[0]]+[hdul[name] for name in names])

    new_hdul.writeto(filename, overwrite=True)
    hdul.close()


prefix = "iter"
new_prefix = "delta_attributes"
filenames = [prefix+".fits.gz"]+[prefix+f"_iteration{i}.fits.gz" for i in range(1, 10)]
new_filenames = [filename.replace(prefix, new_prefix) for filename in filenames]

for filename, new_filename in zip(filenames, new_filenames):
    hdul = fits.open(filename)

    hdul = hdul[:-1]

    del hdul[1].header["PIXORDER"]
    del hdul[1].header["NSIDE"]
    data = hdul[1].data
    for index in range(data["stack"].size):
        if data["stack"][index] != 0.0:
            data["stack"][:index] = data["stack"][index]
            break
    for index in range(data["stack"].size - 1, 0, -1):
        if data["stack"][index] != 0.0:
            data["stack"][index:] = data["stack"][index]
            break
    hdu = fits.BinTableHDU(data, name="STACK_DELTAS", header=hdul[1].header)
    hdul[1] = hdu


    data = np.array(hdul[2].data)[["loglam", "eta", "var_lss", "fudge"]]
    hdu = fits.BinTableHDU(data, name="VAR_FUNC", header=hdul[2].header)
    hdul[2] = hdu

    #data = np.array(hdul[3].data)[["loglam_rest", "mean_cont"]]
    #hdu = fits.BinTableHDU(data, name=hdul[3].name, header=hdul[3].header)
    #hdul[3] = hdu

    hdul.writeto(new_filename, overwrite=True)

    hdul.close()
    os.remove(filename)

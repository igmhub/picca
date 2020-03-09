##Functions used for masking BAL features
##LE 2020


import fitsio
import scipy as sp

def read_bal(fbal): ##Based on read_dla from picca/py/picca/io.py
    lst = ['THING_ID','VMIN_CIV_450','VMAX_CIV_450']
    h = fitsio.FITS(fbal)
    bcat = { k :h[1][k][:] for k in lst }
    h.close()

    return bcat


def add_bal_rf(bcat,thid): ##LE based on add_dla from picca/py/picca/data.py
    ### Store the line wavelengths in Angstroms
    lines = {
        "lCIV" : 1549, #Used for testing
        "lNV" : 1240.81,
        "lLya" : 1216.1,
    }

    vmin_AI = []
    vmax_AI = []

    ##Match thing_id to BAL catalog index
    indx = sp.where(bcat['THING_ID']==thid)[0][0]

    #Get all the min/max velocity pairs
    vminArr = bcat['VMIN_CIV_450'] #velocity in km/s
    for i in vminArr[indx]:
        if i > 0:
            vmin_AI.append(i)

    vmaxArr = bcat['VMAX_CIV_450']
    for i in vmaxArr[indx]:
        if i > 0:
            vmax_AI.append(i)

    BAL_mask_rf = []

    ls = sp.constants.c*10**-3 ##Speed of light in km/s

    ##Calculate mask width for each velocity pair, for each emission line
    for i in range(len(vmin_AI)): 
        for lin in lines.values():
            lMax = lin*(1-vmin_AI[i]/ls)
            lMin = lin*(1-vmax_AI[i]/ls)
            ##add a condition where lMin < Ly-A for speed?
            BAL_mask_rf += [[lMin, lMax]]

    BAL_mask_rf = sp.log10(sp.asarray(BAL_mask_rf))

    return BAL_mask_rf

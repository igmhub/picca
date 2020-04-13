##Functions used for masking BAL features
##LE 2020


import fitsio
import scipy as sp

def read_bal(fbal): ##Based on read_dla from picca/py/picca/io.py
    lst = ['THING_ID','VMIN_CIV_450','VMAX_CIV_450','VMIN_CIV_2000','VMAX_CIV_2000']
    h = fitsio.FITS(fbal)
    bcat = { k :h[1][k][:] for k in lst }
    h.close()

    return bcat


def add_bal_rf(bcat,thid,BALi): ##LE based on add_dla from picca/py/picca/data.py
    ### Store the line wavelengths in Angstroms
    lines = {
        "lCIV" : 1549, #Used for testing
        "lNV" : 1240.81,
        "lLya" : 1216.1,
    }

    if BALi == 'BI':
        lst = ['VMIN_CIV_2000','VMAX_CIV_2000']
    elif BALi == 'BOTH':
        lst = ['VMIN_CIV_450','VMAX_CIV_450','VMIN_CIV_2000','VMAX_CIV_2000']
    else: ##Leaving AI as default
        lst = ['VMIN_CIV_450','VMAX_CIV_450']

    BAL_mask_rf = []

    ls = sp.constants.c*10**-3 ##Speed of light in km/s

    vMin = []
    vMax = []

    ##Match thing_id to BAL catalog index
    indx = sp.where(bcat['THING_ID']==thid)[0][0]

    #Get all the min/max velocity pairs
    for i in lst:
        if i.find('VMIN') == 0: ##Feels risky
            velArr = bcat[i]
            for j in velArr[indx]:
                if j > 0:
                    vMin.append(j)
        else:
            velArr = bcat[i]
            for j in velArr[indx]:
                if j > 0:
                    vMax.append(j)


    ##Calculate mask width for each velocity pair, for each emission line
    for i in range(len(vMin)): 
        for lin in lines.values():
            lMax = lin*(1-vMin[i]/ls)
            lMin = lin*(1-vMax[i]/ls)
            ##Only bother if actually in the forest
            if (lMin < 1216.1): 
                BAL_mask_rf += [[lMin, lMax]]

    BAL_mask_rf = sp.log10(sp.asarray(BAL_mask_rf))

    return BAL_mask_rf

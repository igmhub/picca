##Functions used for masking BAL features ##LE 2020


import fitsio
import scipy as sp

def read_bal(bal_catalog): ##Based on read_dla from picca/py/picca/io.py
    lst = ['THING_ID','VMIN_CIV_450','VMAX_CIV_450','VMIN_CIV_2000','VMAX_CIV_2000']
    h = fitsio.FITS(bal_catalog)
    bal_dict = { k :h[1][k][:] for k in lst }
    h.close()

    return bal_dict


#based on add_dla from picca/py/picca/data.py
def add_bal_rf(bal_catalog,thingid,bal_index): #BAL catalog, THING_ID, and BAL index
    ### Wavelengths in Angstroms
    lines = {
        "lCIV" : 1549, #Used for testing
        "lNV" : 1240.81,
        "lLya" : 1216.1,
        "lLyb" : 1020,
        "lOIV" : 1031,
        "lOVI" : 1037,
        "lOI" : 1039
    }

    if bal_index == 'bi':
        lst = ['VMIN_CIV_2000','VMAX_CIV_2000']
    else: ##AI, the default
        lst = ['VMIN_CIV_450','VMAX_CIV_450']

    bal_mask_rf = []

    ls = sp.constants.c*10**-3 ##Speed of light in km/s

    vMin = [] ##list of minimum velocities
    vMax = [] ##list of maximum velocities

    ##Match thing_id to BAL catalog index
    indx = sp.where(bal_catalog['THING_ID']==thingid)[0][0]

    #Store the min/max velocity pairs from the BAL catalog
    for i in lst:
        if i.find('VMIN') == 0: ##Feels risky
            velocity_list = bal_catalog[i]
            for j in velocity_list[indx]:
                if j > 0:
                    vMin.append(j)
        else:
            velocity_list = bal_catalog[i]
            for j in velocity_list[indx]:
                if j > 0:
                    vMax.append(j)

    ##Calculate mask width for each velocity pair, for each emission line
    for i in range(len(vMin)): 
        for lin in lines.values():
            lMin = lin*(1-vMin[i]/ls)
            lMax = lin*(1-vMax[i]/ls)
            bal_mask_rf += [[lMin, lMax]]
    
    bal_mask_rf = sp.log10(sp.asarray(bal_mask_rf))

    return bal_mask_rf


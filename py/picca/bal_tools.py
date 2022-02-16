""" This module defines functions for masking BAL absorption.

This module provides two functions:
    - read_bal
    - add_bal_rest_frame
See the respective docstrings for more details
"""

import fitsio
import numpy as np
from astropy.table import Table

from . import constants

def read_bal(filename,mode):  ##Based on read_dla from picca/py/picca/io.py
    """Copies just the BAL information from the catalog.

    Args:
        filename: str
            Catalog name
        mode: str
            From args.mode, sets catalog type

    Returns:
        A dictionary with BAL information. Keys are the TARGETID
        associated with the BALs. Values are a tuple with its AI
        (*_CIV_450) and BI (*_CIV_2000) velocity.

    """
    if 'desi' in mode:
        id_name = 'TARGETID'
        ext_name = 'ZCATALOG'
    
    else:
        id_name = 'THING_ID'
        ext_name = 'BALCAT'

    column_list = [
        id_name, 'VMIN_CIV_450', 'VMAX_CIV_450', 'AI_CIV'
    ]

    hdul = fitsio.FITS(filename)
    bal_catalog = {col: hdul[ext_name][col][:] for col in column_list}
    hdul.close()

    return bal_catalog

def add_bal_mask(bal_catalog, objectid, mode):
    """Creates a list of wavelengths to be masked out by forest.mask

    Args:
        bal_catalog: str
            Catalog of BALs
        objectid: str
            Identifier of quasar
        mode: str
            From args.mode, sets catalog type
    """

    if 'desi' in mode:
        id_name = 'TARGETID'
    else:
        id_name = 'THING_ID'

    ### Wavelengths in Angstroms
    lines = {
        "lCIV": 1549,
        "lNV": 1240.81,
        "lLya": 1216.1,
        "lCIII": 1175,
        "lPV1": 1117,
        "lPV2": 1128,
        "lSIV1": 1062,
        "lSIV2": 1074,
        "lLyb": 1020,
        "lOIV": 1031,
        "lOVI": 1037,
        "lOI": 1039
    }

    velocity_list = ['VMIN_CIV_450', 'VMAX_CIV_450']

    light_speed = constants.SPEED_LIGHT

    bal_mask = Table(names=['log_wave_min','log_wave_max','frame'], dtype=['f4','f4','S2'])
    min_velocities = []  ##list of minimum velocities
    max_velocities = []  ##list of maximum velocities

    ##Match objectid of object to BAL catalog index
    match_index = np.where(bal_catalog[id_name] == objectid)[0][0]

    #Store the min/max velocity pairs from the BAL catalog
    for col in velocity_list:
        if col.find('VMIN') == 0:
            velocity_list = bal_catalog[col]
            for vel in velocity_list[match_index]:
                if vel > 0:
                    min_velocities.append(vel)
        else:
            velocity_list = bal_catalog[col]
            for vel in velocity_list[match_index]:
                if vel > 0:
                    max_velocities.append(vel)

    ##Calculate mask width for each velocity pair, for each emission line
    for vel in range(len(min_velocities)):
        for line in lines.values():
            min_wavelength = np.log10(line * (1 - min_velocities[vel] / light_speed))
            max_wavelength = np.log10(line * (1 - max_velocities[vel] / light_speed))
            #VMIN and VMAX were switched between the eBOSS and DESI BAL catalogs.
            if 'desi' in mode:
                bal_mask.add_row([max_wavelength,min_wavelength,'RF'])
            else:
                bal_mask.add_row([min_wavelength,max_wavelength,'RF'])

    return bal_mask

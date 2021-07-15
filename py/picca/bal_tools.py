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


def read_bal(filename):  ##Based on read_dla from picca/py/picca/io.py
    """Reads the BAL catalog from a fits file.

    Args:
        filename: str
            Catalog of BALs

    Returns:
        A dictionary with BAL information. Keys are the THING_ID
        associated with the BALs. Values are a tuple with its AI
        (*_CIV_450) and BI (*_CIV_2000) velocity.

    """
    column_list = [
        'THING_ID', 'VMIN_CIV_450', 'VMAX_CIV_450', 'VMIN_CIV_2000',
        'VMAX_CIV_2000'
    ]
    hdul = fitsio.FITS(filename)
   # bal_dict = {col: hdul['BALCAT'][col][:] for col in column_list}
    bal_dict = {col: hdul[1][col][:] for col in column_list}
    hdul.close()

    return bal_dict


def add_bal_rest_frame(bal_catalog, thingid, bal_index):
    """Creates a list of wavelengths to be masked out by forest.mask

    Args:
        bal_catalog: str
            Catalog of BALs
        thingid: str
            thingid of quasar (eBOSS)
        bal_index: str
            which index to use (AI or BI). In picca_deltas.py, AI is
            used as the default.

    """
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

    if bal_index == 'bi':
        velocity_list = ['VMIN_CIV_2000', 'VMAX_CIV_2000']
    else:  ##AI, the default
        velocity_list = ['VMIN_CIV_450', 'VMAX_CIV_450']


    light_speed = constants.SPEED_LIGHT

    mask_rest_frame_bal = Table(names=['log_wave_min','log_wave_max','frame'], dtype=['f4','f4','S2'])
    min_velocities = []  ##list of minimum velocities
    max_velocities = []  ##list of maximum velocities

    ##Match thing_id of object to BAL catalog index
    match_index = np.where(bal_catalog['THING_ID'] == thingid)[0][0]

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
            mask_rest_frame_bal.add_row([min_wavelength,max_wavelength,'RF'])


    return mask_rest_frame_bal

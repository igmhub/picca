#!/usr/bin/env python
import numpy as np
import sys
import h5py


def get_scan(path, output, chi2_format="chi2"):
    ff = h5py.File(path, 'r')

    # Extract the scan
    scan = ff['chi2 scan']
    pars = scan['result'].attrs
    full_data = np.array(scan['result/values'])

    # Get the columns
    at = full_data.T[pars['at']]
    ap = full_data.T[pars['ap']]
    chi2 = full_data.T[pars['fval']]

    # Figure out if we need likelihood or chi2
    if chi2_format == "lik":
        last_col = np.exp(-0.5 * (chi2 - np.min(chi2)))
    else:
        last_col = chi2

    # Create the output array and sort by alpha perp
    output_data = np.c_[at, ap, last_col]
    output_data = output_data[output_data[:, 0].argsort(kind='mergesort')]

    # Write the output file
    np.savetxt(output, output_data)

    ff.close()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: " + __file__ + " <input_h5_file>" + " <output_file>" +
              " chi2/lik")
        print("The input must be a valid picca fitter2 scan output HDF file.")
        print(
            "At the end add the word <chi2> if you want the last column to be chi2 \
            or <lik> if you want it to be a likelihood")
    else:
        path = sys.argv[1]
        output = sys.argv[2]
        chi2_format = sys.argv[3]
        if chi2_format != "chi2" and chi2_format != "lik":
            raise ValueError("At the end add the word <chi2> if you want the \
            last column to be chi2 or <lik> if you want it to be a likelihood")

        get_scan(path, output, chi2_format)

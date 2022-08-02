"""This module defines the method compute_continuum to compute the quasar continua"""
import iminuit
import numpy as np

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.least_squares.least_squares_cont_model import LeastsSquaresContModel


def compute_continuum(forest, get_mean_cont, get_eta, get_var_lss, get_fudge,
                      use_constant_weight, order):
    """Compute the forest continuum.

    Fits a model based on the mean quasar continuum and linear function
    (see equation 2 of du Mas des Bourboux et al. 2020)
    Flags the forest with bad_cont if the computation fails.

    Arguments
    ---------
    forest: Forest
    A forest instance where the continuum will be computed

    get_mean_cont: scipy.interpolate.interp1d
    Interpolation function to compute the unabsorbed mean quasar continua.

    get_eta: scipy.interpolate.interp1d
    Interpolation function to compute mapping function eta. See equation 4 of
    du Mas des Bourboux et al. 2020 for details.

    get_fudge: scipy.interpolate.interp1d
    Interpolation function to compute mapping function fudge. See equation 4 of
    du Mas des Bourboux et al. 2020 for details.

    get_var_lss: scipy.interpolate.interp1d
    Interpolation function to compute mapping functions var_lss. See equation 4 of
    du Mas des Bourboux et al. 2020 for details.

    use_constant_weight: boolean
    If "True", set all the delta weights to one.

    order: int
    Order of the polynomial for the continuum fit.

    Return
    ------
    cont_model: array of float or None
    The quasar continuum. None if there were problems computing it

    bad_continuum_reason: str or None
    The reason why the continuum could not be computed. None when there were
    no problems

    continuum_fit_parameters: (float, float)
    The zero-point and the slope used in the linear part of the continuum model
    """
    # get mean continuum
    mean_cont = get_mean_cont(forest.log_lambda - np.log10(1 + forest.z))

    # add transmission correction
    # (previously computed using method add_optical_depth)
    mean_cont *= forest.transmission_correction

    mean_cont_kwargs = {"mean_cont": mean_cont}
    # TODO: This can probably be replaced by forest.log_lambda[-1] and
    # forest.log_lambda[0]
    mean_cont_kwargs["log_lambda_max"] = (
        Forest.log_lambda_rest_frame_grid[-1] + np.log10(1 + forest.z))
    mean_cont_kwargs["log_lambda_min"] = (Forest.log_lambda_rest_frame_grid[0] +
                                          np.log10(1 + forest.z))

    #
    weights_kwargs = {
        "use_constant_weight": use_constant_weight,
        "eta": get_eta(forest.log_lambda),
        "var_lss": get_var_lss(forest.log_lambda),
        "fudge": get_fudge(forest.log_lambda),
    }

    leasts_squares = LeastsSquaresContModel(forest=forest,
                                            mean_cont_kwargs=mean_cont_kwargs,
                                            weights_kwargs=weights_kwargs)

    zero_point = (forest.flux * forest.ivar).sum() / forest.ivar.sum()
    slope = 0.0

    minimizer = iminuit.Minuit(leasts_squares,
                               zero_point=zero_point,
                               slope=slope)
    minimizer.errors["zero_point"] = zero_point / 2.
    minimizer.errors["slope"] = zero_point / 2.
    minimizer.errordef = 1.
    minimizer.print_level = 0
    minimizer.fixed["slope"] = order == 0
    minimizer.migrad()

    bad_continuum_reason = None
    cont_model = leasts_squares.get_continuum_model(
        forest, minimizer.values["zero_point"], minimizer.values["slope"],
        **mean_cont_kwargs)
    if not minimizer.valid:
        bad_continuum_reason = "minuit didn't converge"
    if np.any(cont_model < 0):
        bad_continuum_reason = "negative continuum"

    if bad_continuum_reason is None:
        continuum_fit_parameters = (minimizer.values["zero_point"],
                                    minimizer.values["slope"])
    ## if the continuum is negative or minuit didn't converge, then
    ## set it to None
    else:
        cont_model = None
        continuum_fit_parameters = (np.nan, np.nan)

    return cont_model, bad_continuum_reason, continuum_fit_parameters

def _solve_gls(x_matrix, weights, flux):
    xw_matrix = x_matrix.T * weights
    xwx_matrix = xw_matrix @ x_matrix
    yw_vector = xw_matrix @ flux
    poly_coef = np.linalg.solve(xwx_matrix, yw_vector)

    return poly_coef

def _polynomial_sum(log_lambda_slope_arr, coef):
    order = coef.size
    return np.sum([coef[n]*log_lambda_slope_arr**n for n in range(order)],
        axis=0)

# Generalized least squares fitting
# see https://en.wikipedia.org/wiki/Generalized_least_squares
def fit_continuum_gls(forest, get_mean_cont, get_eta, get_var_lss, get_fudge,
                      use_constant_weight, order, num_iter):
    """Compute the forest continuum.

    Fits a model based on the mean quasar continuum and linear function
    (see equation 2 of du Mas des Bourboux et al. 2020)
    Flags the forest with bad_cont if the computation fails.

    Arguments
    ---------
    forest: Forest
    A forest instance where the continuum will be computed

    get_mean_cont: scipy.interpolate.interp1d
    Interpolation function to compute the unabsorbed mean quasar continua.

    get_eta: scipy.interpolate.interp1d
    Interpolation function to compute mapping function eta. See equation 4 of
    du Mas des Bourboux et al. 2020 for details.

    get_fudge: scipy.interpolate.interp1d
    Interpolation function to compute mapping function fudge. See equation 4 of
    du Mas des Bourboux et al. 2020 for details.

    get_var_lss: scipy.interpolate.interp1d
    Interpolation function to compute mapping functions var_lss. See equation 4 of
    du Mas des Bourboux et al. 2020 for details.

    use_constant_weight: boolean
    If "True", set all the delta weights to one.

    order: int
    Order of the polynomial for the continuum fit.

    Return
    ------
    cont_model: array of float or None
    The quasar continuum. None if there were problems computing it

    bad_continuum_reason: str or None
    The reason why the continuum could not be computed. None when there were
    no problems

    continuum_fit_parameters: (float, float)
    The zero-point and the slope used in the linear part of the continuum model
    """
    # get mean continuum
    mean_cont = get_mean_cont(forest.log_lambda - np.log10(1 + forest.z))

    # add transmission correction
    # (previously computed using method add_optical_depth)
    mean_cont *= forest.transmission_correction

    log_lambda_max = forest.log_lambda[-1]
    log_lambda_min = forest.log_lambda[0]
    log_lambda_slope_arr = (forest.log_lambda - log_lambda_min) / (log_lambda_max - log_lambda_min)

    # same as np.column_stack
    x_matrix = np.column_stack(
        [mean_cont*(log_lambda_slope_arr**n) for n in range(order+1)]
    )

    if forest.continuum is None:
        cont_model = mean_cont
    else:
        cont_model = np.where(forest.continuum>0, forest.continuum, 1)

    bad_continuum_reason = None

    # print("=======================")
    poly_coefficients = np.zeros(2)
    for _ in range(num_iter):
        if use_constant_weight:
            weights = np.ones_like(forest.flux)
        else:
            var_pipe = 1. / forest.ivar / cont_model**2
            var_lss = get_var_lss(forest.log_lambda)
            eta = get_eta(forest.log_lambda)
            fudge = get_fudge(forest.log_lambda)
            variance = eta * var_pipe + var_lss + fudge / var_pipe
            weights = 1.0 / cont_model**2 / variance

        try:
            poly_coefficients[:order+1] = _solve_gls(x_matrix, weights, forest.flux)
        except np.linalg.LinAlgError:
            bad_continuum_reason = "singular matrix"
            break

        poly = _polynomial_sum(log_lambda_slope_arr, poly_coefficients)
        cont_model_new = mean_cont * poly
        dcont = cont_model_new - cont_model
        cont_model = cont_model_new
        # print(np.mean(dcont**2))
        if np.all(dcont < 1e-6):
            # print("converged")
            break

    if np.any(cont_model < 0):
        bad_continuum_reason = "negative continuum"

    if bad_continuum_reason is None:
        continuum_fit_parameters = poly
    ## if problem, then set it to None
    else:
        cont_model = None
        continuum_fit_parameters = (np.nan, np.nan)

    return cont_model, bad_continuum_reason, continuum_fit_parameters

def fit_var_stats_gls(var_pipe_values, var_delta, var2_delta):
    if len(var_pipe_values) < 1:
        return None, None
    x_matrix = np.column_stack([var_pipe_values, np.ones_like(var_pipe_values)])
    y_vector = var_delta
    weights = var2_delta

    try:
        solution = _solve_gls(x_matrix, weights, y_vector)
    except np.linalg.LinAlgError:
        bad_continuum_reason = "singular matrix"
        return None, None

    return solution
    

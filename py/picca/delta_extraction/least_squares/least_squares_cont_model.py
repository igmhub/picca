"""This module defines the class LeastsSquaresContModel"""
import numpy as np


class LeastsSquaresContModel:
    """This class deals with the continuum fitting.

    It is passed to iminuit and when called it will return the chi2 for a given
    set of parameters

    Methods
    -------
    __init__
    __call__

    Attributes
    ----------
    forest: Forest
    Forest instance where the model is fit

    expected_flux: Dr16ExpectedFlux
    Dr16ExpectedFlux instance running the fit

    mean_cont_kwargs: dict
    kwargs passed to expected_flux.get_continuum_model

    weights_kwargs: dict
    kwargs passed to expected_flux.get_continuum_weights
    """

    def __init__(self,
                 forest,
                 expected_flux,
                 mean_cont_kwargs=None,
                 weights_kwargs=None):
        """Initialize class instances

        Arguments
        ---------
        forest: Forest
        The forest to fit

        expected_flux: Dr16ExpectedFlux
        The expected flux instance

        mean_cont_kwargs: dict or None - default = None
        kwargs needed by method get_continuum_model of expected_flux. If None
        then it will be assigned an empty dictionary

        weights_kwargs: dict or None - default = None
        kwargs needed by method get_continuum_weights of expected_flux. If None
        then it will be assigned an empty dictionary
        """
        self.forest = forest
        self.expected_flux = expected_flux
        if mean_cont_kwargs is None:
            self.mean_cont_kwargs = {}
        else:
            self.mean_cont_kwargs = mean_cont_kwargs
        if weights_kwargs is None:
            self.weights_kwargs = {}
        else:
            self.weights_kwargs = weights_kwargs

    def __call__(self, zero_point, slope):
        """
        Compute chi2 for a given set of parameters

        Arguments
        ---------
        zero_point: float
        Zero point of the linear function (flux mean). Referred to as $a_q$ in
        du Mas des Bourboux et al. 2020

        slope: float
        Slope of the linear function (evolution of the flux). Referred to as
        $b_q$ in du Mas des Bourboux et al. 2020

        Returns
        -------
        chi2: float
        The chi2 for this run
        """
        cont_model = self.expected_flux.get_continuum_model(
            self.forest, zero_point, slope, **self.mean_cont_kwargs)

        weights = self.expected_flux.get_continuum_weights(
            self.forest, cont_model, **self.weights_kwargs)

        chi2_contribution = (self.forest.flux - cont_model)**2 * weights
        return chi2_contribution.sum() - np.log(weights).sum()

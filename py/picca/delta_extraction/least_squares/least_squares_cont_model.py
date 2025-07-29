"""This module defines the class LeastsSquaresContModel"""
import numpy as np

from picca.delta_extraction.errors import LeastSquaresError


class LeastsSquaresContModel:
    """This class deals with the continuum fitting.

    It is passed to iminuit and when called it will return the chi2 for a given
    set of parameters

    Methods
    -------
    __init__
    __call__
    get_continuum_model

    Attributes
    ----------
    forest: Forest
    Forest instance where the model is fit

    mean_cont_kwargs: dict
    kwargs passed to get_continuum_model

    ndata: int
    Number of datapoints used in the fit

    weights_kwargs: dict
    kwargs passed to get_continuum_weights
    """

    def __init__(self,
                 forest,
                 mean_cont_kwargs=None,
                 weights_kwargs=None):
        """Initialize class instances

        Arguments
        ---------
        forest: Forest
        The forest to fit

        mean_cont_kwargs: dict or None - default = None
        kwargs needed by method get_continuum_model. If None
        then it will be assigned an empty dictionary

        weights_kwargs: dict or None - default = None
        kwargs needed by method get_continuum_weights. If None
        then it will be assigned an empty dictionary
        """
        self.forest = forest
        if mean_cont_kwargs is None:
            self.mean_cont_kwargs = {}
        else:
            self.mean_cont_kwargs = mean_cont_kwargs
        if weights_kwargs is None:
            self.weights_kwargs = {}
        else:
            self.weights_kwargs = weights_kwargs

        self.ndata = None

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
        cont_model = self.get_continuum_model(
            self.forest, zero_point, slope, **self.mean_cont_kwargs)

        weights = self.get_continuum_weights(
            self.forest, cont_model, **self.weights_kwargs)

        w = weights > 0
        chi2_contribution = (self.forest.flux - cont_model)**2 * weights

        if self.ndata is None:
            self.ndata =  self.forest.flux[w].size
        return chi2_contribution.sum() - np.log(weights[w]).sum()

    def get_continuum_model(self, forest, zero_point, slope, **kwargs):
        """Get the model for the continuum fit

        Arguments
        ---------
        forest: Forest
        The forest instance we want the model from

        zero_point: float
        Zero point of the linear function (flux mean). Referred to as $a_q$ in
        du Mas des Bourboux et al. 2020

        slope: float
        Slope of the linear function (evolution of the flux). Referred to as
        $b_q$ in du Mas des Bourboux et al. 2020

        Keyword Arguments
        -----------------
        mean_cont: array of floats
        Mean continuum. Required.

        log_lambda_max: float
        Maximum log_lambda for this forest.

        log_lambda_min: float
        Minimum log_lambda for this forest.

        Return
        ------
        cont_model: array of float
        The continuum model
        """
        # unpack kwargs
        if "mean_cont" not in kwargs:
            raise LeastSquaresError("Function get_continuum_model requires "
                                    "'mean_cont' in the **kwargs dictionary")
        mean_cont = kwargs.get("mean_cont")
        for key in ["log_lambda_max", "log_lambda_min"]:
            if key not in kwargs:
                raise LeastSquaresError("Function get_continuum_model requires "
                                        f"'{key}' in the **kwargs dictionary")
        log_lambda_max = kwargs.get("log_lambda_max")
        log_lambda_min = kwargs.get("log_lambda_min")
        # compute continuum
        line = (slope * (forest.log_lambda - log_lambda_min) /
                (log_lambda_max - log_lambda_min) + zero_point)

        return line * mean_cont

    def get_continuum_weights(self, forest, cont_model, **kwargs):
        """Get the continuum model weights

        Arguments
        ---------
        forest: Forest
        The forest instance we want the model from

        cont_model: array of float
        The continuum model

        Keyword Arguments
        -----------------
        eta: array of floats
        Correction to the pipeline noise (see du Mas des Bourboux et al. 2020)

        var_lss: array of floats
        Contribution of sigma_lss to the total variance (see du Mas des Bourboux
        et al. 2020)

        fudge: array of floats
        Fudge contribution to the total variance. (see du Mas des Bourboux et
        al. 2020)

        Return
        ------
        weights: array of float
        The continuum model weights
        """
        if "use_constant_weight" not in kwargs:
            raise LeastSquaresError("Function get_continuum_weights requires "
                                    "'use_constant_weight' in the **kwargs dictionary")
        # Assign 0 weight to pixels with ivar==0
        w = forest.ivar > 0
        weights = np.empty_like(forest.log_lambda)
        weights[~w] = 0

        if kwargs.get("use_constant_weight"):
            weights[w] = 1
        else:
            for key in ["eta", "var_lss", "fudge"]:
                if key not in kwargs:
                    raise LeastSquaresError("Function get_continuum_weights requires "
                                            f"'{key}' in the **kwargs dictionary")
            var_pipe = 1. / forest.ivar[w] / cont_model[w]**2
            var_lss = kwargs["var_lss"]
            eta = kwargs["eta"]
            fudge = kwargs["fudge"]
            variance = eta[w] * var_pipe + var_lss[w] + fudge[w] / var_pipe
            weights[w] = 1.0 / cont_model[w]**2 / variance

        return weights

    def get_ndata(self):
        """Get the number of datapoints used in the fit

        Return
        ndata: int
        Number of datapoints used in the fit
        """
        return self.ndata

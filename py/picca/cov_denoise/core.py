"""Core routines for denoising Lya forest covariance matrices.

A noisy covariance matrix is denoised in two steps:

1. Its correlation matrix is projected onto a reference eigenbasis V (from a
   clean, high-precision reference). The diagonal of the projection (the
   pseudo-eigenvalues) is used to build an initial reconstruction.
2. The pseudo-eigenvalues are classified against groups of training mocks
   (e.g. mock types). The group clean residuals, weighted by inverse-squared
   classification distance, define a residual eigenbasis U. The noisy residual
   is projected onto U and reconstructed from its diagonal, then added back.

The scripts picca_denoise_fit_classifier.py and picca_denoise_pipeline.py run
the two phases at scale. DenoiseModel applies a fitted model to single matrices.
"""
import os

import numpy as np
from scipy.linalg.blas import dgemm, dsymm
from scipy.ndimage import gaussian_filter1d

from picca.utils import userprint
from picca.cov_denoise.io import load_sample_paths, read_cov


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------

def corr_matrix(cov):
    """Convert a covariance matrix to a correlation matrix."""
    var = np.diagonal(cov)
    return cov / np.sqrt(var * var[:, None])


def compute_reference_eigenbasis(reference_matrix):
    """Eigenvectors of the reference matrix, sorted by decreasing eigenvalue."""
    eigenvalues, eigvecs = np.linalg.eigh(reference_matrix)
    return eigvecs[:, np.argsort(eigenvalues)[::-1]]


def project(matrix, basis):
    """Return basis^T @ matrix @ basis for symmetric `matrix`."""
    tmp = dsymm(alpha=1.0, a=matrix, b=basis, side=0, lower=0)
    return dgemm(alpha=1.0, a=basis, b=tmp, trans_a=True)


def reconstruct_from_pseudoeigvals(pev, basis):
    """Rebuild a correlation matrix from pseudo-eigenvalues (unit diagonal)."""
    scaled = basis * pev[np.newaxis, :]
    recon = dgemm(alpha=1.0, a=scaled, b=basis, trans_b=True)
    np.fill_diagonal(recon, 1.0)
    return recon


def diag_total_power(projected):
    """Fraction of the Frobenius power of `projected` on its diagonal."""
    diag_power = np.sum(np.diag(projected) ** 2)
    total_power = np.linalg.norm(projected, "fro") ** 2
    return diag_power / total_power


def smooth_pev_adaptive(pev, sigma_logk=0.05):
    """Smooth pseudo-eigenvalues with a Gaussian of fixed width in log(k).

    The effective window therefore grows with mode number k.
    """
    n = len(pev)
    k = np.arange(1, n + 1)
    logk = np.log10(k)
    logk_uniform = np.linspace(logk[0], logk[-1], n)
    pev_logk = np.interp(logk_uniform, logk, pev)
    pev_logk_smooth = gaussian_filter1d(
        pev_logk, sigma=sigma_logk * n / (logk[-1] - logk[0]))
    return np.interp(logk, logk_uniform, pev_logk_smooth)


def kl_divergence(cov_true, cov_approx, n):
    """KL divergence between N(0, cov_true) and N(0, cov_approx).

    Returns -1.0 if cov_approx is singular or the determinant of
    cov_approx^-1 @ cov_true is non-positive (i.e. cov_approx is not
    positive definite).
    """
    try:
        matprod = np.linalg.inv(cov_approx) @ cov_true
        trace_term = np.trace(matprod)
        sign, logdet_term = np.linalg.slogdet(matprod)
        if sign <= 0:
            return -1.0
        return 0.5 * (trace_term - n - logdet_term)
    except np.linalg.LinAlgError:
        return -1.0


def fix_positive_definite(corr, force=False):
    """Clip eigenvalues to 1e-8 * max eigenvalue and rebuild the matrix.

    Args:
        corr: correlation matrix
        force: if False, return `corr` unchanged when its smallest eigenvalue
            is non-negative. If True, always rebuild.

    Returns:
        (fixed_corr, was_fixed, n_negative, min_eigenvalue)
    """
    eigvals, eigvecs = np.linalg.eigh(corr)
    n_neg = int(np.sum(eigvals < 0))
    min_eig = float(eigvals[0])
    if not force and min_eig >= 0:
        return corr, False, n_neg, min_eig
    eigvals = np.maximum(eigvals, 1e-8 * eigvals.max())
    return reconstruct_from_pseudoeigvals(eigvals, eigvecs), True, n_neg, min_eig


def scale_to_cov(corr, var):
    """Convert a correlation matrix back to covariance with variances `var`."""
    return corr * np.sqrt(var * var[:, None])


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def weighted_distance(pev, clf):
    """Feature-weighted z-score distance of `pev` to each group of `clf`."""
    distances = {}
    for key in clf.unique_keys:
        z2 = ((pev - clf.group_mean[key]) / clf.group_std[key]) ** 2
        distances[key] = float(np.sqrt(np.sum(clf.weights * z2)))
    return distances


def inv_sq_weights(distances):
    """Normalized inverse-squared-distance weights."""
    inv_sq = {k: 1.0 / (d ** 2 + 1e-12) for k, d in distances.items()}
    total = sum(inv_sq.values())
    return {k: v / total for k, v in inv_sq.items()}


class GroupClassifier:
    """Weighted diagonal z-score classifier in pseudo-eigenvalue space."""

    def __init__(self):
        self.unique_keys = None
        self.n_features = None
        self.group_mean = None
        self.group_std = None
        self.weights = None

    def fit(self, pevs, keys):
        """Fit group means/stds and feature weights.

        Feature weights are the between/within group variance ratio times the
        mean |pseudo-eigenvalue|, which up-weights leading modes that carry
        the dominant correlation structure. Weights sum to 1.
        """
        self.unique_keys = sorted(set(keys))
        self.n_features = pevs.shape[1]
        keys_arr = np.array(keys, dtype=object)
        masks = {k: np.array([ki == k for ki in keys_arr])
                 for k in self.unique_keys}

        self.group_mean = {}
        self.group_std = {}
        for key in self.unique_keys:
            self.group_mean[key] = pevs[masks[key]].mean(axis=0)
            self.group_std[key] = pevs[masks[key]].std(axis=0) + 1e-12

        group_means_pev = np.array([self.group_mean[k] for k in self.unique_keys])
        between_var = group_means_pev.var(axis=0)
        within_var = np.zeros(self.n_features)
        for key in self.unique_keys:
            within_var += pevs[masks[key]].var(axis=0) / len(self.unique_keys)

        pev_mean = np.abs(pevs.mean(axis=0))
        raw_weights = between_var / (within_var + 1e-12) * pev_mean
        self.weights = raw_weights / (raw_weights.sum() + 1e-12)
        return self

    def predict_hard(self, pevs):
        """Nearest-group label for each row of `pevs`."""
        results = []
        for pev in pevs:
            dists = weighted_distance(pev, self)
            results.append(min(dists, key=dists.get))
        return results

    def to_arrays(self):
        """Arrays for saving into a model .npz file."""
        arrays = {
            "clf_weights": self.weights,
            "clf_n_features": np.array(self.n_features),
        }
        for key in self.unique_keys:
            arrays[f"clf_mean__{key}"] = self.group_mean[key]
            arrays[f"clf_std__{key}"] = self.group_std[key]
        return arrays

    @classmethod
    def from_model_data(cls, model_data):
        """Load the classifier from an opened model .npz."""
        clf = cls()
        clf.unique_keys = [str(k) for k in model_data["unique_keys"]]
        clf.weights = model_data["clf_weights"]
        clf.n_features = int(model_data["clf_n_features"])
        clf.group_mean = {k: model_data[f"clf_mean__{k}"] for k in clf.unique_keys}
        clf.group_std = {k: model_data[f"clf_std__{k}"] for k in clf.unique_keys}
        return clf


# ---------------------------------------------------------------------------
# Pipeline steps (shared by the scripts and DenoiseModel)
# ---------------------------------------------------------------------------

def compute_clean_residual(clean_corr, basis):
    """Residual of a clean correlation matrix w.r.t. its own reconstruction."""
    pev_clean = np.diag(project(clean_corr, basis)).copy()
    return clean_corr - reconstruct_from_pseudoeigvals(pev_clean, basis)


def initial_reconstruction(noisy_corr, basis, smooth_pev):
    """Step 1: project onto the reference basis and reconstruct.

    Returns:
        pev, pev_smooth (a copy of pev if smoothing is off), power_ratio,
        initial_recon
    """
    proj = project(noisy_corr, basis)
    pev = np.diag(proj).copy()
    power_ratio = diag_total_power(proj)
    pev_smooth = smooth_pev_adaptive(pev) if smooth_pev else pev.copy()
    initial_recon = reconstruct_from_pseudoeigvals(pev_smooth, basis)
    return pev, pev_smooth, power_ratio, initial_recon


def residual_correction(noisy_corr, initial_recon, pev_for_clf, clf,
                        clean_resid, smooth_pev):
    """Step 2: classify, build the residual basis, correct the residual.

    Returns:
        dict with distances, weights, resid_pev, resid_pev_smooth,
        resid_power_ratio and final_corr
    """
    distances = weighted_distance(pev_for_clf, clf)
    weights = inv_sq_weights(distances)

    resid_ref = sum(weights[k] * clean_resid[k] for k in clf.unique_keys)
    eigvals, resid_basis = np.linalg.eigh(resid_ref)
    resid_basis = resid_basis[:, np.argsort(np.abs(eigvals))[::-1]]

    resid_proj = project(noisy_corr - initial_recon, resid_basis)
    resid_power_ratio = diag_total_power(resid_proj)
    resid_pev = np.diag(resid_proj).copy()
    if smooth_pev:
        resid_pev_smooth = smooth_pev_adaptive(resid_pev)
        resid_pev_use = resid_pev_smooth
    else:
        resid_pev_smooth = resid_pev.copy()
        resid_pev_use = resid_pev
    scaled = resid_basis * resid_pev_use[np.newaxis, :]
    resid_recon = dgemm(alpha=1.0, a=scaled, b=resid_basis, trans_b=True)

    final_corr = initial_recon + resid_recon
    np.fill_diagonal(final_corr, 1.0)
    return {
        "distances": distances,
        "weights": weights,
        "resid_pev": resid_pev,
        "resid_pev_smooth": resid_pev_smooth,
        "resid_power_ratio": resid_power_ratio,
        "final_corr": final_corr,
    }


# ---------------------------------------------------------------------------
# Model file
# ---------------------------------------------------------------------------

def save_model(path, basis, clf, clean_resid, metadata):
    """Save basis, classifier, clean residuals and run metadata to .npz."""
    if not path.endswith(".npz"):
        path += ".npz"
    arrays = {"V": basis, "unique_keys": np.array(clf.unique_keys)}
    arrays.update(clf.to_arrays())
    for key in clf.unique_keys:
        arrays[f"resid__{key}"] = clean_resid[key]
    for name, value in metadata.items():
        arrays[f"meta__{name}"] = np.array(value)
    np.savez(path, **arrays)
    return path


def read_model_metadata(model_data):
    """Metadata stored by save_model (empty dict for older model files)."""
    return {name[len("meta__"):]: model_data[name]
            for name in model_data.files if name.startswith("meta__")}


# ---------------------------------------------------------------------------
# Notebook-facing model
# ---------------------------------------------------------------------------

class DenoiseModel:
    """A fitted denoising model, for applying to individual matrices.

    Usage:
        model = DenoiseModel.load_sample("DR1")   # model path from the config
        model = DenoiseModel.load("denoise_model.npz")
        result = model.denoise(noisy_cov)             # (n, n) array
        result = model.denoise("/path/to/noisy.fits")
        result = model.denoise(noisy_cov, truth_cov="/path/to/truth.fits")
    """

    def __init__(self, basis, clf, clean_resid, smooth_pev=False):
        self.V = basis
        self.clf = clf
        self.unique_keys = clf.unique_keys
        self.clean_resid = clean_resid
        self.smooth_pev = smooth_pev
        self.n = basis.shape[0]

    @classmethod
    def load(cls, model_path):
        """Load a model .npz written by picca_denoise_fit_classifier.py."""
        # allow_pickle for model files written by the pre-picca scripts
        model_data = np.load(model_path, allow_pickle=True)
        clf = GroupClassifier.from_model_data(model_data)
        clean_resid = {k: model_data[f"resid__{k}"] for k in clf.unique_keys}
        metadata = read_model_metadata(model_data)
        smooth_pev = bool(metadata.get("smooth_pev", False))
        userprint(f"Model loaded: n={model_data['V'].shape[0]}, "
                  f"groups={clf.unique_keys}, smooth_pev={smooth_pev}")
        return cls(model_data["V"], clf, clean_resid, smooth_pev)

    @classmethod
    def load_sample(cls, sample, paths_config=None):
        """Load the model listed for `sample` in the paths config.

        Args:
            sample: sample name, e.g. 'DR1'
            paths_config: optional .ini layered on top of default_paths.ini
        """
        sample_paths = load_sample_paths(sample, paths_config)
        model_path = sample_paths["model"]
        if model_path is None:
            raise ValueError(
                f"No model listed for sample {sample_paths['sample']}. Fit one "
                "with picca_denoise_fit_classifier.py and set 'model' in the "
                f"[{sample_paths['sample']}] section, or use "
                "DenoiseModel.load(path).")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model for sample {sample_paths['sample']} not found: "
                f"{model_path}")
        return cls.load(model_path)

    @staticmethod
    def _load_cov(cov_input):
        if isinstance(cov_input, str):
            return read_cov(cov_input)
        return np.asarray(cov_input, dtype=np.float64)

    def denoise(self, noisy_cov, truth_cov=None, smooth_pev=None):
        """Denoise one covariance matrix.

        Args:
            noisy_cov: (n, n) covariance array or path to a FITS file
            truth_cov: optional (n, n) covariance array or FITS path, used to
                compute KL divergence, Frobenius norm and MSE diagnostics
            smooth_pev: smooth pseudo-eigenvalues in log-k. Defaults to the
                setting the model was fitted with.

        Returns:
            dict with keys pev, pev_smooth, resid_pev, resid_pev_smooth,
            power_ratio, resid_power_ratio, initial_recon_cov, final_cov,
            distances, weights, hard_pred, npd_corrected_initial,
            npd_corrected_final, KL_initial, KL_final, frob_initial,
            frob_final, mse_initial, mse_final (diagnostics are None
            without truth_cov)
        """
        if smooth_pev is None:
            smooth_pev = self.smooth_pev

        cov = self._load_cov(noisy_cov)
        noisy_var = np.diagonal(cov).copy()
        noisy_corr = corr_matrix(cov)

        pev, pev_smooth, power_ratio, initial_recon = initial_reconstruction(
            noisy_corr, self.V, smooth_pev)
        initial_recon, npd_initial, n_neg, min_eig = fix_positive_definite(
            initial_recon)
        if npd_initial:
            userprint(f"WARNING: non-positive definite initial recon: {n_neg} "
                      f"eigenvalue(s) shifted, lowest = {min_eig:.6f}")
        initial_recon_cov = scale_to_cov(initial_recon, noisy_var)

        corrected = residual_correction(noisy_corr, initial_recon, pev_smooth,
                                        self.clf, self.clean_resid, smooth_pev)
        final_corr, npd_final, n_neg, min_eig = fix_positive_definite(
            corrected["final_corr"])
        if npd_final:
            userprint(f"WARNING: non-positive definite final corr: {n_neg} "
                      f"eigenvalue(s) shifted, lowest = {min_eig:.6f}")
        final_cov = scale_to_cov(final_corr, noisy_var)

        result = {
            "pev": pev,
            "pev_smooth": pev_smooth,
            "resid_pev": corrected["resid_pev"],
            "resid_pev_smooth": corrected["resid_pev_smooth"],
            "power_ratio": power_ratio,
            "resid_power_ratio": corrected["resid_power_ratio"],
            "initial_recon_cov": initial_recon_cov,
            "final_cov": final_cov,
            "distances": corrected["distances"],
            "weights": corrected["weights"],
            "hard_pred": min(corrected["distances"],
                             key=corrected["distances"].get),
            "npd_corrected_initial": int(npd_initial),
            "npd_corrected_final": int(npd_final),
            "KL_initial": None, "KL_final": None,
            "frob_initial": None, "frob_final": None,
            "mse_initial": None, "mse_final": None,
        }

        if truth_cov is not None:
            truth = self._load_cov(truth_cov)
            truth_corr = corr_matrix(truth)
            result["KL_initial"] = kl_divergence(truth, initial_recon_cov, self.n)
            result["KL_final"] = kl_divergence(truth, final_cov, self.n)
            for label, corr in (("initial", initial_recon), ("final", final_corr)):
                diff = corr - truth_corr
                result[f"frob_{label}"] = float(np.linalg.norm(diff, "fro"))
                result[f"mse_{label}"] = float(np.mean(diff ** 2))
        return result

"""Denoising of Lya forest covariance matrices estimated from mocks.

See picca.cov_denoise.core for the method and the scripts
picca_denoise_fit_classifier.py and picca_denoise_pipeline.py.
"""
from picca.cov_denoise.core import DenoiseModel, GroupClassifier
from picca.cov_denoise.io import load_sample_paths, split_train_test

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
from vivarium.framework.randomness import get_hash


def get_random_variable(distribution, seed: str, key=""):
    """Get a random variable from a scipy.stats distribution"""
    np.random.seed(get_hash(f"{key}_{seed}"))
    return distribution.rvs()


def get_norm_from_quantiles(mean: float, lower: float, upper: float,
                            quantiles: Tuple[float, float] = (0.025, 0.975)) -> stats.norm:
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    return stats.norm(loc=mean, scale=sd)


def get_truncnorm_from_quantiles(mean: float, lower: float, upper: float,
                                 quantiles: Tuple[float, float] = (0.025, 0.975),
                                 lower_clip: float = 0.0, upper_clip: float = 1.0) -> stats.truncnorm:
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    try:
        a = (lower_clip - mean) / sd
        b = (upper_clip - mean) / sd
        return stats.truncnorm(loc=mean, scale=sd, a=a, b=b)
    except ZeroDivisionError:
        # degenerate case: if upper == lower, then use the mean with sd==0
        return stats.norm(loc=mean, scale=sd)


def get_lognorm_from_quantiles(mean: float, lower: float, upper: float,
                               quantiles: Tuple[float, float] = (0.025, 0.975)) -> stats.lognorm:
    """Returns a frozen lognormal distribution with the specified mean, such that
    (lower, upper) are approximately equal to the quantiles with ranks
    (quantile_ranks[0], quantile_ranks[1]).
    """
    # Let Y ~ norm(mu, sigma^2) and X = exp(Y), where mu = log(mean)
    # so X ~ lognorm(s=sigma, scale=exp(mu)) in scipy's notation.
    # We will determine sigma from the two specified quantiles lower and upper.
    if not (lower <= mean <= upper):
        raise ValueError(
            f"The mean ({mean}) must be between the lower ({lower}) and upper ({upper}) "
            "quantile values."
        )
    try:
        # mean (and mean) of the normal random variable Y = log(X)
        mu = np.log(mean)
        # quantiles of the standard normal distribution corresponding to quantile_ranks
        stdnorm_quantiles = stats.norm.ppf(quantiles)
        # quantiles of Y = log(X) corresponding to the quantiles (lower, upper) for X
        norm_quantiles = np.log([lower, upper])
        # standard deviation of Y = log(X) computed from the above quantiles for Y
        # and the corresponding standard normal quantiles
        sigma = (norm_quantiles[1] - norm_quantiles[0]) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
        # Frozen lognormal distribution for X = exp(Y)
        # (s=sigma is the shape parameter; the scale parameter is exp(mu), which equals the mean)
        return stats.lognorm(s=sigma, scale=mean)
    except:
        return stats.norm(loc=mean, scale=0)


def generate_vectorized_lognormal_draws(df: pd.DataFrame, seed: str,) -> pd.DataFrame:
    mean = df['mean_value'].values
    lower = df['lower_value'].values
    upper = df['upper_value'].values
    assert np.all((lower <= mean) & (mean <= upper))
    assert np.all((lower == mean) == (upper == mean))

    sample_mask = (mean > 0) & (lower < mean) & (mean < upper)
    stdnorm_quantiles = stats.norm.ppf((0.025, 0.975))
    norm_quantiles = np.log([lower[sample_mask], upper[sample_mask]])
    sigma = (norm_quantiles[1] - norm_quantiles[0]) / (
                stdnorm_quantiles[1] - stdnorm_quantiles[0])

    distribution = stats.lognorm(s=sigma, scale=mean[sample_mask])
    np.random.seed(get_hash(seed))
    lognorm_samples = distribution.rvs(size=(1000, sample_mask.sum())).T
    lognorm_samples = pd.DataFrame(lognorm_samples, index=df[sample_mask].index)

    use_means = np.tile(mean[~sample_mask], 1000).reshape((1000, ~sample_mask.sum())).T
    use_means = pd.DataFrame(use_means, index=df[~sample_mask].index)
    draws = pd.concat([lognorm_samples, use_means])
    draws = draws.sort_index().rename(columns=lambda d: f'draw_{d}')
    return draws

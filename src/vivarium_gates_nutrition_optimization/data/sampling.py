import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple
from vivarium.framework.randomness import get_hash

def get_norm_from_quantiles(mean: float, lower: float, upper: float,
                            quantiles: Tuple[float, float] = (0.025, 0.975)) -> stats.norm:
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    return stats.norm(loc=mean, scale=sd)

def generate_vectorized_lognormal_draws(
    df: pd.DataFrame,
    seed: str,
) -> pd.DataFrame:
    mean = df["mean_value"].values
    lower = df["lower_value"].values
    upper = df["upper_value"].values
    assert np.all((lower <= mean) & (mean <= upper))
    assert np.all((lower == mean) == (upper == mean))
    assert np.all(mean >= 0)

    sample_mask = (mean > 0) & (lower < mean) & (mean < upper)
    stdnorm_quantiles = stats.norm.ppf((0.025, 0.975))
    norm_quantiles = np.log([lower[sample_mask], upper[sample_mask]])
    sigma = (norm_quantiles[1] - norm_quantiles[0]) / (
        stdnorm_quantiles[1] - stdnorm_quantiles[0]
    )

    distribution = stats.lognorm(s=sigma, scale=mean[sample_mask])
    np.random.seed(get_hash(seed))
    lognorm_samples = distribution.rvs(size=(1000, sample_mask.sum())).T
    lognorm_samples = pd.DataFrame(lognorm_samples, index=df[sample_mask].index)

    use_means = np.tile(mean[~sample_mask], 1000).reshape((1000, ~sample_mask.sum())).T
    use_means = pd.DataFrame(use_means, index=df[~sample_mask].index)
    draws = pd.concat([lognorm_samples, use_means])
    draws = draws.sort_index().rename(columns=lambda d: f"draw_{d}")
    return draws

from typing import Union

import pandas as pd
from gbd_mapping import ModelableEntity, causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey


def get_entity(key: Union[str, EntityKey]) -> ModelableEntity:
    key = EntityKey(key)
    # Map of entity types to their gbd mappings.
    type_map = {
        "cause": causes,
        "covariate": covariates,
        "risk_factor": risk_factors,
    }
    return type_map[key.type][key.name]


def expand_draw_columns(data: pd.DataFrame, num_draws: int, num_repeats: int) -> pd.DataFrame:
    """
    Expands draw columns in the input DataFrame by repeating them num_repeats times.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing draw columns.
        num_draws (int): Number of draw columns to expand (e.g., 100).
        num_repeats (int): Number of times to repeat the draw columns (e.g., 5).

    Returns:
        pd.DataFrame: DataFrame with expanded draw columns.
    """
    draw_cols = [f"draw_{i}" for i in range(num_draws)]
    expanded_draws = []

    for i in range(num_repeats):
        df_copy = data[draw_cols].copy()
        df_copy.columns = [f"draw_{j}" for j in range(i * num_draws, (i + 1) * num_draws)]
        expanded_draws.append(df_copy)
    expanded_draws_df = pd.concat(expanded_draws, axis=1)

    return expanded_draws_df

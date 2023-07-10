from typing import Union

from gbd_mapping import (
    causes,
    covariates,
    risk_factors,
    ModelableEntity,
)
from vivarium.framework.artifact import EntityKey


def get_entity(key: Union[str, EntityKey]) -> ModelableEntity:
    key = EntityKey(key)
    # Map of entity types to their gbd mappings.
    type_map = {
        'cause': causes,
        'covariate': covariates,
        'risk_factor': risk_factors,
    }
    return type_map[key.type][key.name]

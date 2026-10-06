import pandas as pd
from vivarium.gbd_mapping import sequelae
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import utilities as vi_utils
from vivarium_gbd_access.gbd import base_data as gbd
from vivarium_gbd_access.gbd.demographics import get_age_group_id
from vivarium_gbd_access.gbd.measures import get_birth_exposure, get_relative_risk
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data

from vivarium_gates_nutrition_optimization.constants import data_keys
from vivarium_gates_nutrition_optimization.data import utilities

_ALL_SEXES = gbd_constants.SEX.MALE + gbd_constants.SEX.FEMALE


@vi_utils.cache
def load_2021_lbwsg_birth_exposure(location: str) -> pd.DataFrame:
    """Pull the GBD 2021 LBWSG birth exposure draws for a location."""
    entity = utilities.get_entity(data_keys.LBWSG.EXPOSURE)
    location_id = utility_data.resolve_location(location)
    data = get_birth_exposure(
        entity.gbd_id,
        location_id,
        2022,
        "draws",
        release_id=gbd_constants.RELEASE_IDS.GBD_2021,
    )
    return data


@vi_utils.cache
def get_maternal_disorder_ylds(location: str) -> pd.DataFrame:
    """Pull GBD 2023 maternal disorders YLD rate draws for a location."""
    entity = utilities.get_entity(data_keys.MATERNAL_DISORDERS.YLDS)
    location_id = utility_data.resolve_location(location)
    data = gbd.get_machinery_estimates(
        entity="cause",
        entity_id=int(entity.gbd_id),
        release_id=gbd_constants.RELEASE_IDS.GBD_2023,
        estimates="draws",
        measure_id=vi_globals.MEASURES["YLDs"],
        metric_id=vi_globals.METRICS["Rate"],
        location_id=location_id,
        sex_id=_ALL_SEXES,
        age_group_id=get_age_group_id(gbd_constants.RELEASE_IDS.GBD_2023),
        year_id=2023,
    )
    return data


@vi_utils.cache
def get_anemia_ylds(location: str) -> pd.DataFrame:
    """Pull GBD 2023 YLD rate draws for the maternal hemorrhage anemia sequelae."""
    anemia_sequelae = [
        sequelae.mild_anemia_due_to_maternal_hemorrhage,
        sequelae.moderate_anemia_due_to_maternal_hemorrhage,
        sequelae.severe_anemia_due_to_maternal_hemorrhage,
    ]
    location_id = utility_data.resolve_location(location)
    # One call per sequela: get_machinery_estimates is only known to accept a scalar
    # entity_id.
    data = pd.concat(
        [
            gbd.get_machinery_estimates(
                entity="sequela",
                entity_id=int(sequela.gbd_id),
                release_id=gbd_constants.RELEASE_IDS.GBD_2023,
                estimates="draws",
                measure_id=vi_globals.MEASURES["YLDs"],
                metric_id=vi_globals.METRICS["Rate"],
                location_id=location_id,
                sex_id=_ALL_SEXES,
                age_group_id=get_age_group_id(gbd_constants.RELEASE_IDS.GBD_2023),
                year_id=2023,
            )
            for sequela in anemia_sequelae
        ],
        ignore_index=True,
    )
    return data


def get_hbg_less_than_70(location: str) -> pd.DataFrame:
    """Pull the proportion of pregnant women with hemoglobin below 70 g/L."""
    raise NotImplementedError(
        "get_hbg_less_than_70 (rei 207 prevalence, release 33) needs the gbd_access 7 "
        "translation gated on the MIC-7505 release-33 spike."
    )


## This is left as 2021 because the changes to hemoglobin RRs are very significant and would require other model updates we don't plan to make.
@vi_utils.cache
def get_hemoglobin_maternal_disorders_rr() -> pd.DataFrame:
    """Relative risk associated with one g/dL decrease in hemoglobin concentration below 12 g/dL"""
    data = get_relative_risk(
        95,
        location_id=1,
        year_id=2021,
        data_type="draws",
        release_id=gbd_constants.RELEASE_IDS.GBD_2021,
    )
    # Subset to a single sub-cause as the call returns values for 10 sub-causes within the
    # maternal disorders parent cause
    data = data[(data["cause_id"] == 367) & (data["sex_id"].isin(gbd_constants.SEX.FEMALE))]
    return data


def get_hemoglobin_exposure_data(key: str, location: str) -> pd.DataFrame:
    """Get hemoglobin exposure mean or standard deviation draws for a location."""
    raise NotImplementedError(
        "get_hemoglobin_exposure_data (rei 376 exposure and exposure SD, release 33) needs "
        "the gbd_access 7 translation gated on the MIC-7505 release-33 spike."
    )

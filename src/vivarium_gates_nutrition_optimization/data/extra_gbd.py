from pathlib import Path

import pandas as pd
from vivarium.gbd_mapping import sequelae
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import utilities as vi_utils
from vivarium_gbd_access.gbd import base_data as gbd
from vivarium_gbd_access.gbd.demographics import get_age_group_id
from vivarium_gbd_access.gbd.measures import (
    get_birth_exposure,
    get_exposure,
    get_relative_risk,
)
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data

from vivarium_gates_nutrition_optimization.constants import data_keys
from vivarium_gates_nutrition_optimization.data import utilities

_ALL_SEXES = gbd_constants.SEX.MALE + gbd_constants.SEX.FEMALE
# The hemoglobin models this model was built on live under release 33, not the
# GBD 2023 compare release the other pulls use.
HEMOGLOBIN_RELEASE_ID = 33
HEMOGLOBIN_REI_ID = 376
HEMOGLOBIN_BELOW_70_REI_ID = 207
ALL_CAUSES_ID = 294
PREGNANCY_AGE_GROUP_IDS = list(range(7, 16))  # 10-14 through 50-54 years
HEMOGLOBIN_YEAR_ID = 2023

# Central comp will not publish the release-33 hemoglobin SD model (ME 28870, model
# version 926443) for ihme_cc_get_estimates, so it is read from the draws they copied
# to the shared folder instead. See the README there and MIC-7505.
HEMOGLOBIN_RELEASE_33_DIR = Path("/mnt/team/simulation_science/pub/data/hemoglobin_release33")
HEMOGLOBIN_SD_FILES = [
    "hemoglobin_exposure_sd_ethiopia.parquet",
    "hemoglobin_exposure_sd_nigeria.parquet",
    "hemoglobin_exposure_sd_pakistan.parquet",
    "me_28870_mvid_926443_select_locations.parquet",
]
HEMOGLOBIN_SD_ID_COLUMNS = [
    "location_id",
    "year_id",
    "age_group_id",
    "sex_id",
    "measure_id",
    "metric_id",
    "modelable_entity_id",
    "rei_id",
]


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
    data = gbd.get_machinery_estimates(
        entity="sequela",
        entity_id=[int(sequela.gbd_id) for sequela in anemia_sequelae],
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
def get_hbg_less_than_70(location: str) -> pd.DataFrame:
    """Pull the proportion of pregnant women with hemoglobin below 70 g/L."""
    location_id = utility_data.resolve_location(location)
    data = gbd.get_machinery_estimates(
        entity="impairment-cause",
        entity_id=[(HEMOGLOBIN_BELOW_70_REI_ID, ALL_CAUSES_ID)],
        release_id=HEMOGLOBIN_RELEASE_ID,
        estimates="draws",
        measure_id=vi_globals.MEASURES["Prevalence"],
        metric_id=vi_globals.METRICS["Rate"],
        location_id=location_id,
        sex_id=gbd_constants.SEX.FEMALE,
        age_group_id=PREGNANCY_AGE_GROUP_IDS,
        year_id=2023,
    )
    return data


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
    """Get hemoglobin exposure mean or standard deviation draws for a location.

    The mean is a live release-33 pull. The standard deviation is read from the
    stored release-33 capture, because GBD does not serve that model version
    through the shared functions.
    """
    location_id = utility_data.resolve_location(location)
    if key == data_keys.HEMOGLOBIN.STANDARD_DEVIATION:
        return _read_release_33_hemoglobin_sd(location_id)
    return get_exposure(
        HEMOGLOBIN_REI_ID,
        location_id,
        HEMOGLOBIN_YEAR_ID,
        "draws",
        sex_id=gbd_constants.SEX.FEMALE,
        release_id=HEMOGLOBIN_RELEASE_ID,
    )


def _read_release_33_hemoglobin_sd(location_id: int) -> pd.DataFrame:
    """Read the stored release-33 hemoglobin SD draws for one location."""
    frames = []
    for name in HEMOGLOBIN_SD_FILES:
        path = HEMOGLOBIN_RELEASE_33_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. See the README in {HEMOGLOBIN_RELEASE_33_DIR} "
                "and MIC-7505."
            )
        frames.append(pd.read_parquet(path))
    data = pd.concat(frames, ignore_index=True)
    data = data[
        (data["location_id"] == location_id)
        & data["sex_id"].isin(gbd_constants.SEX.FEMALE)
        & (data["year_id"] == HEMOGLOBIN_YEAR_ID)
    ]
    if data.empty:
        raise FileNotFoundError(
            f"No release-33 hemoglobin SD draws stored for location {location_id} in "
            f"{HEMOGLOBIN_RELEASE_33_DIR}; ask central comp to add it (MIC-7505)."
        )
    draw_columns = [column for column in data.columns if column.startswith("draw_")]
    data = data.assign(rei_id=HEMOGLOBIN_REI_ID)
    return data[HEMOGLOBIN_SD_ID_COLUMNS + draw_columns].reset_index(drop=True)

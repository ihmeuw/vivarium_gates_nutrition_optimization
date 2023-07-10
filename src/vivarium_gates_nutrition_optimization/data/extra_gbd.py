from gbd_mapping import sequelae
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import gbd
from vivarium_gbd_access import utilities as vi_utils
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data

from vivarium_gates_nutrition_optimization.constants import data_keys
from vivarium_gates_nutrition_optimization.data import utilities

GBD_2020_ROUND_ID = 7


@gbd.memory.cache
def load_lbwsg_exposure(location: str):
    entity = utilities.get_entity(data_keys.LBWSG.EXPOSURE)
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        gbd_id_type="rei_id",
        gbd_id=entity.gbd_id,
        source=gbd_constants.SOURCES.EXPOSURE,
        location_id=location_id,
        sex_id=gbd_constants.SEX.MALE + gbd_constants.SEX.FEMALE,
        age_group_id=164,  # Birth prevalence
        gbd_round_id=gbd_constants.ROUND_IDS.GBD_2019,
        decomp_step=gbd_constants.DECOMP_STEP.STEP_4,
    )
    # This data set is big, so let's reduce it by a factor of ~40
    data = data[data["year_id"] == 2019].drop(columns="year_id")
    return data


@gbd.memory.cache
def get_maternal_disorder_ylds(location: str):
    entity = utilities.get_entity(data_keys.MATERNAL_DISORDERS.YLDS)
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        "cause_id",
        entity.gbd_id,
        source=gbd_constants.SOURCES.COMO,
        location_id=location_id,
        decomp_step=gbd_constants.DECOMP_STEP.STEP_5,
        gbd_round_id=gbd_constants.ROUND_IDS.GBD_2019,
        measure_id=vi_globals.MEASURES["YLDs"],
    )
    return data


@gbd.memory.cache
def get_anemia_ylds(location: str):
    anemia_sequelae = [
        sequelae.mild_anemia_due_to_maternal_hemorrhage,
        sequelae.moderate_anemia_due_to_maternal_hemorrhage,
        sequelae.severe_anemia_due_to_maternal_hemorrhage,
    ]
    anemia_ids = [s.gbd_id for s in anemia_sequelae]
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        "sequela_id",
        anemia_ids,
        source=gbd_constants.SOURCES.COMO,
        location_id=location_id,
        decomp_step=gbd_constants.DECOMP_STEP.STEP_5,
        gbd_round_id=gbd_constants.ROUND_IDS.GBD_2019,
        measure_id=vi_globals.MEASURES["YLDs"],
    )
    return data

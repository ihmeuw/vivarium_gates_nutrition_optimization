from gbd_mapping import sequelae
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import gbd
from vivarium_gbd_access import utilities as vi_utils
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data

from vivarium_gates_nutrition_optimization.constants import data_keys
from vivarium_gates_nutrition_optimization.data import utilities


@gbd.memory.cache
def load_lbwsg_exposure(location: str):
    entity = utilities.get_entity(data_keys.LBWSG.EXPOSURE)
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        gbd_id_type="rei_id",
        gbd_id=entity.gbd_id,
        source=gbd_constants.SOURCES.EXPOSURE,
        location_id=location_id,
        year_id=2021,
        sex_id=gbd_constants.SEX.MALE + gbd_constants.SEX.FEMALE,
        age_group_id=164,  # Birth prevalence
        release_id = gbd_constants.RELEASE_IDS.GBD_2021,
    )
    return data


@gbd.memory.cache
def get_all_cause_yld_rate(location: str):
    entity = utilities.get_entity("cause.all_causes.ylds")
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        "cause_id",
        entity.gbd_id,
        source=gbd_constants.SOURCES.COMO,
        location_id=location_id,
        release_id = gbd_constants.RELEASE_IDS.GBD_2021,
        measure_id=vi_globals.MEASURES["YLDs"],
        metric_id=3,  # rate
    )
    return data


@gbd.memory.cache
def get_maternal_disorder_ylds(location: str, metric_id=None):
    entity = utilities.get_entity(data_keys.MATERNAL_DISORDERS.YLDS)
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        "cause_id",
        entity.gbd_id,
        source=gbd_constants.SOURCES.COMO,
        location_id=location_id,
        release_id = gbd_constants.RELEASE_IDS.GBD_2021,
        measure_id=vi_globals.MEASURES["YLDs"],
        metric_id=metric_id,
    )
    return data


@gbd.memory.cache
def get_anemia_ylds(location: str, metric_id=None):
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
        release_id = gbd_constants.RELEASE_IDS.GBD_2021,
        measure_id=vi_globals.MEASURES["YLDs"],
        metric_id=metric_id,
    )
    return data


@gbd.memory.cache
def get_anemia_yld_rate(location: str):
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        "rei_id",
        192,
        source=gbd_constants.SOURCES.COMO,
        location_id=location_id,
        release_id = gbd_constants.RELEASE_IDS.GBD_2021,
        measure_id=vi_globals.MEASURES["YLDs"],
        metric_id=3,
    )
    return data


@gbd.memory.cache
def get_hemoglobin_maternal_disorders_rr():
    """Relative risk associated with one g/dL decrease in hemoglobin concentration below 12 g/dL"""
    data = vi_utils.get_draws(
        gbd_id_type="rei_id",
        gbd_id=95,
        release_id = gbd_constants.RELEASE_IDS.GBD_2021,
        year_id=2021,
        sex_id=2,
        source="rr",
        status="best",
    )
    # Subset to a single sub-cause as the get_draws call returns values for 10 sub-causes within the
    # maternal disorders parent cause
    data = data[data["cause_id"] == 367]
    return data

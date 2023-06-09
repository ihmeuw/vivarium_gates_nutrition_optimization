"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.
"""
from typing import Union

import pandas as pd
from gbd_mapping import causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import gbd
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import interface
from vivarium_inputs import utilities as vi_utils
from vivarium_inputs import utility_data
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_gates_nutrition_optimization.constants import data_keys, metadata
from vivarium_gates_nutrition_optimization.data import extra_gbd, sampling

def get_data(lookup_key: str, location: str) -> pd.DataFrame:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        data_keys.POPULATION.LOCATION: load_population_location,
        data_keys.POPULATION.STRUCTURE: load_population_structure,
        data_keys.POPULATION.AGE_BINS: load_age_bins,
        data_keys.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        data_keys.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        data_keys.POPULATION.ACMR: load_standard_data,
        data_keys.PREGNANCY.ASFR: load_asfr,
        data_keys.PREGNANCY.SBR: load_sbr,
        data_keys.PREGNANCY.LIVE_BIRTHS_BY_SEX: load_standard_data,
        data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE: load_standard_data,
        data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC: load_standard_data,
    }
    return mapping[lookup_key](lookup_key, location)


def load_population_location(key: str, location: str) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f"Unrecognized key {key}")

    return location


def load_population_structure(key: str, location: str) -> pd.DataFrame:
    return interface.get_population_structure(location)


def load_age_bins(key: str, location: str) -> pd.DataFrame:
    return interface.get_age_bins()


def load_demographic_dimensions(key: str, location: str) -> pd.DataFrame:
    return interface.get_demographic_dimensions(location)


def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)
    return interface.get_measure(entity, key.measure, location).droplevel("location")


def load_metadata(key: str, location: str):
    key = EntityKey(key)
    entity = get_entity(key)
    entity_metadata = entity[key.measure]
    if hasattr(entity_metadata, "to_dict"):
        entity_metadata = entity_metadata.to_dict()
    return entity_metadata


def load_categorical_paf(key: str, location: str) -> pd.DataFrame:
    try:
        risk = {
            # todo add keys as needed
            data_keys.KEYGROUP.PAF: data_keys.KEYGROUP,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    distribution_type = get_data(risk.DISTRIBUTION, location)

    if distribution_type != "dichotomous" and "polytomous" not in distribution_type:
        raise NotImplementedError(
            f"Unrecognized distribution {distribution_type} for {risk.name}. Only dichotomous and "
            f"polytomous are recognized categorical distributions."
        )

    exp = get_data(risk.EXPOSURE, location)
    rr = get_data(risk.RELATIVE_RISK, location)

    # paf = (sum_categories(exp * rr) - 1) / sum_categories(exp * rr)
    sum_exp_x_rr = (
        (exp * rr)
        .groupby(list(set(rr.index.names) - {"parameter"}))
        .sum()
        .reset_index()
        .set_index(rr.index.names[:-1])
    )
    paf = (sum_exp_x_rr - 1) / sum_exp_x_rr
    return paf


def _load_em_from_meid(location, meid, measure):
    location_id = utility_data.get_location_id(location)
    data = gbd.get_modelable_entity_draws(meid, location_id)
    data = data[data.measure_id == vi_globals.MEASURES[measure]]
    data = vi_utils.normalize(data, fill_value=0)
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS)
    data = vi_utils.reshape(data)
    data = vi_utils.scrub_gbd_conventions(data, location)
    data = vi_utils.split_interval(data, interval_column="age", split_column_prefix="age")
    data = vi_utils.split_interval(data, interval_column="year", split_column_prefix="year")
    return vi_utils.sort_hierarchical_data(data).droplevel("location")


##################
# Pregnancy Data #
##################

def get_pregnancy_end_rate(location: str):
    asfr = get_data(data_keys.PREGNANCY.ASFR, location)
    sbr = get_data(data_keys.PREGNANCY.SBR, location)
    incidence_c995 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE, location)
    incidence_c374 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC, location)
    pregnancy_end_rate = (asfr + asfr * sbr + incidence_c995 + incidence_c374)
    return pregnancy_end_rate.reorder_levels(asfr.index.names)


def load_asfr(key: str, location: str):
    asfr = load_standard_data(key, location)
    asfr = asfr.reset_index()
    asfr_pivot = asfr.pivot(
        index=[col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"],
        columns='parameter',
        values='value'
    )
    seed = f'{key}_{location}'
    asfr_draws = sampling.generate_vectorized_lognormal_draws(asfr_pivot, seed)
    return asfr_draws


def load_sbr(key: str, location: str):

    births_per_location_year, sbr = [], []
    for child_loc in get_child_locs(location):
        child_pop = get_data(data_keys.POPULATION.STRUCTURE, child_loc)
        child_asfr = get_data(data_keys.PREGNANCY.ASFR, child_loc)
        child_asfr.index = child_pop.index  # Add location back
        child_births = (child_asfr
                        .multiply(child_pop.value, axis=0)
                        .groupby(['location', 'year_start'])
                        .sum())
        births_per_location_year.append(child_births)

        try:
            child_sbr = get_child_sbr(child_loc)
        except vi_globals.DataDoesNotExistError:
            pass

        child_sbr = (child_sbr
                     .reset_index(level='year_end', drop=True)
                     .reindex(child_births.index, level='year_start'))
        sbr.append(child_sbr)

    births_per_location_year = pd.concat(births_per_location_year)
    sbr = pd.concat(sbr)

    births_per_year = births_per_location_year.groupby('year_start').transform('sum')
    sbr = (births_per_location_year
           .multiply(sbr.value, axis=0)
           .divide(births_per_year)
           .groupby('year_start')
           .sum()
           .reset_index())
    sbr['year_end'] = sbr['year_start'] + 1
    sbr = sbr.set_index(['year_start', 'year_end'])
    return sbr

def get_child_sbr(location: str):
    child_sbr = load_standard_data(data_keys.PREGNANCY.SBR, location)
    child_sbr = (child_sbr
                 .reorder_levels(['parameter', 'year_start', 'year_end'])
                 .loc['mean_value'])
    return child_sbr

def get_entity(key: Union[str, EntityKey]):
    # Map of entity types to their gbd mappings.
    type_map = {
        "cause": causes,
        "covariate": covariates,
        "risk_factor": risk_factors,
        "alternative_risk_factor": alternative_risk_factors,
    }
    key = EntityKey(key)
    return type_map[key.type][key.name]


###########
# Helpers #
###########

def get_child_locs(location):
    parent_id = utility_data.get_location_id(location)
    hierarchy = extra_gbd.get_gbd_hierarchy()

    is_child_loc = hierarchy.path_to_top_parent.str.contains(f',{parent_id},')
    is_country = hierarchy.location_type == "admin0"
    child_locs = hierarchy.loc[is_child_loc & is_country, 'location_name'].tolist()

    # Return just location if location is admin 0 or more granular
    if len(child_locs) == 0:
        child_locs.append(location)
    return child_locs

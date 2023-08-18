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

import numpy as np
import pandas as pd
import scipy.stats
import vivarium_inputs.validation.sim as validation
from vivarium.framework.artifact import EntityKey
from vivarium.framework.randomness import get_hash
from vivarium_gbd_access import gbd
from vivarium_inputs import core as vi_core
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import interface
from vivarium_inputs import utilities as vi_utils
from vivarium_inputs import utility_data

from vivarium_gates_nutrition_optimization.constants import (
    data_keys,
    data_values,
    metadata,
    models,
    paths,
)
from vivarium_gates_nutrition_optimization.data import extra_gbd, sampling
from vivarium_gates_nutrition_optimization.data.utilities import get_entity
from vivarium_gates_nutrition_optimization.utilities import get_random_variable_draws


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
        data_keys.PREGNANCY.ASFR: load_asfr,
        data_keys.PREGNANCY.SBR: load_sbr,
        data_keys.PREGNANCY.RAW_INCIDENCE_RATE_MISCARRIAGE: load_raw_incidence_data,
        data_keys.PREGNANCY.RAW_INCIDENCE_RATE_ECTOPIC: load_raw_incidence_data,
        data_keys.LBWSG.DISTRIBUTION: load_metadata,
        data_keys.LBWSG.CATEGORIES: load_metadata,
        data_keys.LBWSG.EXPOSURE: load_lbwsg_exposure,
        data_keys.MATERNAL_DISORDERS.RAW_INCIDENCE_RATE: load_raw_incidence_data,
        data_keys.MATERNAL_DISORDERS.CSMR: load_maternal_csmr,
        data_keys.MATERNAL_DISORDERS.MORTALITY_PROBABILITY: load_maternal_disorders_mortality_probability,
        data_keys.MATERNAL_DISORDERS.INCIDENT_PROBABILITY: load_pregnant_maternal_disorders_incidence,
        data_keys.MATERNAL_DISORDERS.YLDS: load_maternal_disorders_ylds,
        data_keys.MATERNAL_DISORDERS.RR_ATTRIBUTABLE_TO_HEMOGLOBIN: load_hemoglobin_maternal_disorders_rr,
        data_keys.MATERNAL_DISORDERS.PAF_ATTRIBUTABLE_TO_HEMOGLOBIN: load_hemoglobin_maternal_disorders_paf,
        data_keys.MATERNAL_HEMORRHAGE.RAW_INCIDENCE_RATE: load_raw_incidence_data,
        data_keys.MATERNAL_HEMORRHAGE.CSMR: load_maternal_csmr,
        data_keys.MATERNAL_HEMORRHAGE.INCIDENT_PROBABILITY: load_pregnant_maternal_hemorrhage_incidence,
        data_keys.MATERNAL_HEMORRHAGE.RR_ATTRIBUTABLE_TO_HEMOGLOBIN: load_hemoglobin_maternal_hemorrhage_rr,
        data_keys.MATERNAL_HEMORRHAGE.PAF_ATTRIBUTABLE_TO_HEMOGLOBIN: load_hemoglobin_maternal_hemorrhage_paf,
        data_keys.MATERNAL_HEMORRHAGE.MODERATE_HEMORRHAGE_PROBABILITY: get_moderate_hemorrhage_probability,
        data_keys.HEMOGLOBIN.MEAN: get_hemoglobin_data,
        data_keys.HEMOGLOBIN.STANDARD_DEVIATION: get_hemoglobin_data,
        data_keys.HEMOGLOBIN.PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70: get_hemoglobin_csv_data,
        data_keys.MATERNAL_BMI.PREVALENCE_LOW_BMI_ANEMIC: load_bmi_prevalence,
        data_keys.MATERNAL_BMI.PREVALENCE_LOW_BMI_NON_ANEMIC: load_bmi_prevalence,
        data_keys.MATERNAL_INTERVENTIONS.IFA_COVERAGE: load_ifa_coverage,
        data_keys.MATERNAL_INTERVENTIONS.MMS_STILLBIRTH_RR: load_supplementation_stillbirth_rr,
        data_keys.MATERNAL_INTERVENTIONS.BEP_STILLBIRTH_RR: load_supplementation_stillbirth_rr,
    }
    return mapping[lookup_key](lookup_key, location)


def load_population_location(key: str, location: str) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f"Unrecognized key {key}")

    return location


def load_population_structure(key: str, location: str) -> pd.DataFrame:
    base_population_structure = interface.get_population_structure(location)
    pregnancy_end_rate_avg = get_pregnancy_end_incidence(location)
    pregnant_population_structure = (
        pregnancy_end_rate_avg.multiply(base_population_structure["value"], axis=0)
        .assign(location=location)
        .set_index("location", append=True)
    )
    return vi_utils.sort_hierarchical_data(pregnant_population_structure)


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


# TODO: Remove this if/ when Vivarium Inputs implements the change directly
def load_raw_incidence_data(key: str, location: str) -> pd.DataFrame:
    """Temporary function to short circuit around validation issues in Vivarium Inputs"""
    key = EntityKey(key)
    entity = get_entity(key)
    data = vi_core.get_data(entity, key.measure, location)
    data = vi_utils.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, entity, "incidence_rate", location)
    data = vi_utils.split_interval(data, interval_column="age", split_column_prefix="age")
    data = vi_utils.split_interval(data, interval_column="year", split_column_prefix="year")
    return vi_utils.sort_hierarchical_data(data).droplevel("location")


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


##################
# Pregnancy Data #
##################


def get_pregnancy_end_incidence(location: str) -> pd.DataFrame:
    asfr = get_data(data_keys.PREGNANCY.ASFR, location)
    sbr = get_data(data_keys.PREGNANCY.SBR, location)
    sbr = sbr.reset_index(level="year_end", drop=True).reindex(asfr.index, level="year_start")
    incidence_c995 = get_data(data_keys.PREGNANCY.RAW_INCIDENCE_RATE_MISCARRIAGE, location)
    incidence_c374 = get_data(data_keys.PREGNANCY.RAW_INCIDENCE_RATE_ECTOPIC, location)
    pregnancy_end_rate = (
        asfr + asfr.multiply(sbr["value"], axis=0) + incidence_c995 + incidence_c374
    )
    return pregnancy_end_rate.reorder_levels(asfr.index.names)


def load_asfr(key: str, location: str) -> pd.DataFrame:
    asfr = load_standard_data(key, location)
    asfr = asfr.reset_index()
    asfr_pivot = asfr.pivot(
        index=[col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"],
        columns="parameter",
        values="value",
    )
    seed = f"{key}_{location}"
    asfr_draws = sampling.generate_vectorized_lognormal_draws(asfr_pivot, seed)
    return asfr_draws


def load_sbr(key: str, location: str) -> pd.DataFrame:
    sbr = load_standard_data(key, location)
    sbr = sbr.reorder_levels(["parameter", "year_start", "year_end"]).loc["mean_value"]
    return sbr


##############
# LBWSG Data #
##############


def load_lbwsg_exposure(key: str, location: str) -> pd.DataFrame:
    entity = get_entity(data_keys.LBWSG.EXPOSURE)
    data = extra_gbd.load_lbwsg_exposure(location)
    # This category was a mistake in GBD 2019, so drop.
    extra_residual_category = vi_globals.EXTRA_RESIDUAL_CATEGORY[entity.name]
    data = data.loc[data["parameter"] != extra_residual_category]
    idx_cols = ["location_id", "sex_id", "parameter"]
    data = data.set_index(idx_cols)[vi_globals.DRAW_COLUMNS]

    # Sometimes there are data values on the order of 10e-300 that cause
    # floating point headaches, so clip everything to reasonable values
    data = data.clip(lower=vi_globals.MINIMUM_EXPOSURE_VALUE)

    # normalize so all categories sum to 1
    total_exposure = data.groupby(["location_id", "sex_id"]).transform("sum")
    data = (data / total_exposure).reset_index()
    data = reshape_to_vivarium_format(data, location)
    return data


###########################
# Maternal Disorders Data #
###########################


def load_maternal_csmr(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)
    entity.restrictions.yll_age_group_id_end = 15
    return interface.get_measure(entity, key.measure, location).droplevel("location")


def load_maternal_disorders_ylds(key: str, location: str) -> pd.DataFrame:
    groupby_cols = ["age_group_id", "sex_id", "year_id"]
    draw_cols = [f"draw_{i}" for i in range(1000)]

    all_md_ylds = extra_gbd.get_maternal_disorder_ylds(location)
    all_md_ylds = all_md_ylds[groupby_cols + draw_cols]
    all_md_ylds = reshape_to_vivarium_format(all_md_ylds, location)

    anemia_ylds = extra_gbd.get_anemia_ylds(location)
    anemia_ylds = anemia_ylds.groupby(groupby_cols)[draw_cols].sum().reset_index()
    anemia_ylds = reshape_to_vivarium_format(anemia_ylds, location)

    csmr = get_data(data_keys.MATERNAL_DISORDERS.CSMR, location)
    incidence = load_raw_incidence_data(
        data_keys.MATERNAL_DISORDERS.RAW_INCIDENCE_RATE, location
    )
    idx_cols = incidence.index.names
    incidence = incidence.reset_index()
    #   Update incidence for 55-59 year age group to match 50-54 year age group
    to_duplicate = incidence.loc[(incidence.sex == "Female") & (incidence.age_start == 50.0)]
    to_duplicate["age_start"] = 55.0
    to_duplicate["age_end"] = 60.0
    to_drop = incidence.loc[(incidence.sex == "Female") & (incidence.age_start == 55.0)]
    incidence = (
        pd.concat([incidence.drop(to_drop.index), to_duplicate])
        .set_index(idx_cols)
        .sort_index()
    )

    return (all_md_ylds - anemia_ylds) / (incidence - csmr)


def load_pregnant_maternal_disorders_incidence(key: str, location: str):
    total_incidence = get_data(data_keys.MATERNAL_DISORDERS.RAW_INCIDENCE_RATE, location)
    pregnancy_end_rate = get_pregnancy_end_incidence(location)
    maternal_disorders_incidence = total_incidence / pregnancy_end_rate
    ## We have to normalize, since this comes to a probability with some values > 1
    return maternal_disorders_incidence.applymap(lambda value: 1 if value > 1 else value)


def load_maternal_disorders_mortality_probability(key: str, location: str):
    total_csmr = get_data(data_keys.MATERNAL_DISORDERS.CSMR, location)
    total_incidence = get_data(data_keys.MATERNAL_DISORDERS.RAW_INCIDENCE_RATE, location)
    return total_csmr / total_incidence


def load_pregnant_maternal_hemorrhage_incidence(key: str, location: str):
    mh_incidence = get_data(data_keys.MATERNAL_HEMORRHAGE.RAW_INCIDENCE_RATE, location)
    mh_csmr = get_data(data_keys.MATERNAL_HEMORRHAGE.CSMR, location)
    pregnancy_end_rate = get_pregnancy_end_incidence(location)
    maternal_hemorrhage_incidence = (mh_incidence - mh_csmr) / pregnancy_end_rate
    ## I'm not as sure we need to normalize here, but we may as well.
    return maternal_hemorrhage_incidence.applymap(lambda value: 1 if value > 1 else value)


def load_hemoglobin_maternal_hemorrhage_rr(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.MATERNAL_HEMORRHAGE.RR_ATTRIBUTABLE_TO_HEMOGLOBIN:
        raise ValueError(f"Unrecognized key {key}")

    distribution = data_values.RR_MATERNAL_HEMORRHAGE_ATTRIBUTABLE_TO_HEMOGLOBIN
    dist = sampling.get_lognorm_from_quantiles(*distribution)
    # Get a DataFrame with the desired index
    demographic_dimensions = get_data(data_keys.POPULATION.DEMOGRAPHY, location)

    rng = np.random.default_rng(get_hash(f"{key}_{location}"))
    maternal_hemorrhage_rr = pd.DataFrame(
        np.tile(dist.rvs(size=1000, random_state=rng), (len(demographic_dimensions), 1)),
        columns=vi_globals.DRAW_COLUMNS,
        index=demographic_dimensions.index,
    )
    return maternal_hemorrhage_rr


def load_hemoglobin_maternal_hemorrhage_paf(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.MATERNAL_HEMORRHAGE.PAF_ATTRIBUTABLE_TO_HEMOGLOBIN:
        raise ValueError(f"Unrecognized key {key}")

    rr = get_data(data_keys.MATERNAL_HEMORRHAGE.RR_ATTRIBUTABLE_TO_HEMOGLOBIN, location)
    proportion = get_data(
        data_keys.HEMOGLOBIN.PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70, location
    )

    return (rr * proportion + (1 - proportion) - 1) / (rr * proportion + (1 - proportion))


def load_hemoglobin_maternal_disorders_rr(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.MATERNAL_DISORDERS.RR_ATTRIBUTABLE_TO_HEMOGLOBIN:
        raise ValueError(f"Unrecognized key {key}")

    groupby_cols = ["age_group_id", "sex_id", "year_id"]
    draw_cols = [f"draw_{i}" for i in range(1000)]
    rr = extra_gbd.get_hemoglobin_maternal_disorders_rr()
    rr = rr.groupby(groupby_cols)[draw_cols].sum().reset_index()
    rr = reshape_to_vivarium_format(rr, location)
    return rr


def load_hemoglobin_maternal_disorders_paf(key: str, location: str) -> pd.DataFrame:
    location_id = utility_data.get_location_id(location)
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)

    data = pd.read_csv(paths.HEMOGLOBIN_MATERNAL_DISORDERS_PAF_CSV)
    data = data.set_index("location_id").loc[location_id]
    age_bins = utility_data.get_age_bins()
    data = data.merge(age_bins, on="age_group_id")
    data.draw = data.draw.apply(lambda d: f"draw_{d}")
    data = data.pivot(index=["age_start", "age_end"], columns="draw", values="paf")
    data = data.reset_index(level="age_end", drop=True).reindex(
        demography.index, level="age_start", fill_value=0.0
    )
    return data


def get_moderate_hemorrhage_probability(key: str, location: str) -> pd.DataFrame:
    hemorrhage_dist_params = data_values.PROBABILITY_MODERATE_MATERNAL_HEMORRHAGE
    # Clip a bit higher than zero to avoid underflow error
    dist = sampling.get_truncnorm_from_quantiles(*hemorrhage_dist_params, lower_clip=0.1)
    # random seed
    rng = np.random.default_rng(get_hash(f"hemorrhage_severity"))
    moderate_hemorrhage_probability = pd.DataFrame(
        [dist.rvs(size=1000, random_state=rng)],
        columns=vi_globals.DRAW_COLUMNS,
        index=["probability"],
    )

    return moderate_hemorrhage_probability


###########################
# Hemoglobin Data         #
###########################


def get_hemoglobin_data(key: str, location: str) -> pd.DataFrame:
    me_id = {
        data_keys.HEMOGLOBIN.MEAN: 10487,
        data_keys.HEMOGLOBIN.STANDARD_DEVIATION: 10488,
    }[key]
    correction_factors = data_values.PREGNANCY_CORRECTION_FACTORS[key]

    location_id = utility_data.get_location_id(location)
    hemoglobin_data = gbd.get_modelable_entity_draws(me_id=me_id, location_id=location_id)
    hemoglobin_data = reshape_to_vivarium_format(hemoglobin_data, location)

    return hemoglobin_data * correction_factors


def get_hemoglobin_csv_data(key: str, location: str):
    location_id = utility_data.get_location_id(location)
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)

    data = pd.read_csv(paths.PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70_CSV)
    data = data.set_index("location_id").loc[location_id]
    age_bins = utility_data.get_age_bins()
    data = data.merge(age_bins, on="age_group_id")
    data = data.pivot(index=["age_start", "age_end"], columns="input_draw", values="value")
    data = data.reset_index(level="age_end", drop=True).reindex(
        demography.index, level="age_start", fill_value=0.0
    )
    return data


################
# Maternal BMI #
################


def load_bmi_prevalence(key: str, location: str):
    location_id = utility_data.get_location_id(location)
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)

    path = {
        data_keys.MATERNAL_BMI.PREVALENCE_LOW_BMI_ANEMIC: paths.PREVALENCE_LOW_BMI_ANEMIC_CSV,
        data_keys.MATERNAL_BMI.PREVALENCE_LOW_BMI_NON_ANEMIC: paths.PREVALENCE_LOW_BMI_NON_ANEMIC_CSV,
    }[key]
    data = pd.read_csv(path)

    data = (
        data.set_index(["location_id", "age_group_id", "draw"])
        .loc[location_id]
        .value.unstack()
    )
    data = vi_utils.scrub_gbd_conventions(data, location)
    data = vi_utils.split_interval(data, interval_column="age", split_column_prefix="age")
    data.index = data.index.droplevel(["location", "age_end"])
    data = data.reindex(demography.index, level="age_start").fillna(0.0)

    data = vi_utils.sort_hierarchical_data(data)
    data.index = data.index.droplevel("location")

    return data


##########################
# Maternal interventions #
##########################


def load_ifa_coverage(key: str, location: str) -> pd.DataFrame:
    df = pd.read_csv(
        paths.CSV_RAW_DATA_ROOT / "baseline_ifa_coverage" / (location + ".csv"), index_col=0
    )
    df = df.drop(columns=["location_id", "location_name"]).set_index(["draw"]).T
    return df


def load_ifa_effect_size(key: str, location: str) -> pd.DataFrame:
    loc, scale = data_values.IFA_EFFECT_SIZE
    dist = stats.norm(loc, scale)
    rng = np.random.default_rng(get_hash(f"ifa_effect_size_{location}"))
    ifa_effect_size = pd.DataFrame(
        [dist.rvs(size=1000, random_state=rng)],
        columns=vi_globals.DRAW_COLUMNS,
        index=["value"],
    )
    return ifa_effect_size


def load_supplementation_stillbirth_rr(key: str, location: str) -> pd.DataFrame:
    try:
        distribution = data_values.INTERVENTION_STILLBIRTH_RRS[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    dist = sampling.get_lognorm_from_quantiles(*distribution)
    # Don't hash on key because we want simulants to have the same percentile
    # for MMS RR as for BEP
    rng = np.random.default_rng(get_hash(f"stillbirth_rr_{location}"))
    stillbirth_rr = pd.DataFrame(
        [dist.rvs(size=1000, random_state=rng)],
        columns=vi_globals.DRAW_COLUMNS,
        index=["relative_risk"],
    )
    return stillbirth_rr


def load_supplementation_stillbirth_rr(key: str, location: str) -> pd.DataFrame:
    try:
        distribution = data_values.INTERVENTION_STILLBIRTH_RRS[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    dist = sampling.get_lognorm_from_quantiles(*distribution)
    # Don't hash on key because we want simulants to have the same percentile
    # for MMS RR as for BEP
    rng = np.random.default_rng(get_hash(f"stillbirth_rr_{location}"))
    stillbirth_rr = pd.DataFrame(
        [dist.rvs(size=1000, random_state=rng)],
        columns=vi_globals.DRAW_COLUMNS,
        index=["relative_risk"],
    )
    return stillbirth_rr


##############
#   Helpers  #
##############


def reshape_to_vivarium_format(df, location):
    df = vi_utils.reshape(df, value_cols=vi_globals.DRAW_COLUMNS)
    df = vi_utils.scrub_gbd_conventions(df, location)
    df = vi_utils.split_interval(df, interval_column="age", split_column_prefix="age")
    df = vi_utils.split_interval(df, interval_column="year", split_column_prefix="year")
    df = vi_utils.sort_hierarchical_data(df)
    df.index = df.index.droplevel("location")
    return df

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RESIDUAL_CHOICE
from vivarium_public_health.causal_factor.calibration_constant import (
    get_calibration_constant_pipeline_name,
)

from vivarium_gates_nutrition_optimization.constants import (
    data_keys,
    data_values,
    models,
)
from vivarium_gates_nutrition_optimization.constants.data_values import (
    ANEMIA_DISABILITY_WEIGHTS,
    ANEMIA_THRESHOLD_DATA,
    HEMOGLOBIN_DISTRIBUTION_PARAMETERS,
    HEMOGLOBIN_SCALE_FACTOR_MODERATE_HEMORRHAGE,
    HEMOGLOBIN_SCALE_FACTOR_SEVERE_HEMORRHAGE,
    RR_SCALAR,
    SEVERE_ANEMIA_AMONG_PREGNANT_WOMEN_THRESHOLD,
    TMREL_HEMOGLOBIN_ON_MATERNAL_DISORDERS,
)


class Hemoglobin(Component):
    """
    class for hemoglobin utilities and calculations that in turn will
    be used to find anemia status for simulants.
    """

    @property
    def configuration_defaults(self) -> Dict[str, Dict[str, Any]]:
        return {
            self.name: {
                "data_sources": {
                    "hemorrhage_relative_risk": data_keys.MATERNAL_HEMORRHAGE.RR_ATTRIBUTABLE_TO_HEMOGLOBIN,
                    "maternal_disorders_relative_risk": data_keys.MATERNAL_DISORDERS.RR_ATTRIBUTABLE_TO_HEMOGLOBIN,
                }
            }
        }

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)

        index_columns = [
            "sex",
            "age_start",
            "age_end",
            "year_start",
            "year_end",
        ]

        # load data
        mean = (
            builder.data.load(data_keys.HEMOGLOBIN.MEAN)
            .set_index(index_columns)["value"]
            .rename("mean")
        )
        stddev = (
            builder.data.load(data_keys.HEMOGLOBIN.STANDARD_DEVIATION)
            .set_index(index_columns)["value"]
            .rename("stddev")
        )

        distribution_parameters = self.build_lookup_table(
            builder,
            "hemoglobin_distribution_parameters",
            data_source=pd.concat([mean, stddev], axis=1).reset_index(),
            value_columns=["mean", "stddev"],
        )
        self.maternal_hemorrhage_paf_data = self.get_data(
            builder, "risk_factor.hemoglobin_on_maternal_hemorrhage.paf"
        )
        self.hemorrhage_relative_risk = self.build_lookup_table(
            builder, "hemorrhage_relative_risk"
        )
        self.maternal_disorders_relative_risk = self.build_lookup_table(
            builder, "maternal_disorders_relative_risk"
        )
        self.maternal_disorders_paf_data = self.get_data(
            builder, "risk_factor.hemoglobin_on_maternal_disorder.paf"
        )

        self.moderate_hemorrhage_probability = builder.data.load(
            data_keys.MATERNAL_HEMORRHAGE.MODERATE_HEMORRHAGE_PROBABILITY
        ).value.values[0]

        builder.value.register_attribute_producer(
            "hemoglobin.exposure_parameters", source=distribution_parameters
        )

        # Fix resource dependency cycle
        builder.value.register_attribute_producer(
            "raw_hemoglobin.exposure",
            source=self.hemoglobin_source,
            required_resources=[
                "hemoglobin.exposure_parameters",
                "hemoglobin_distribution_propensity",
                "hemoglobin_percentile",
            ],
        )

        builder.value.register_attribute_producer(
            "hemoglobin.exposure",
            source=["raw_hemoglobin.exposure"],
        )

        builder.value.register_attribute_modifier(
            "maternal_disorders.transition_proportion",
            self.adjust_maternal_disorder_proportion,
            required_resources=["hemoglobin.exposure", self.maternal_disorders_relative_risk],
        )
        builder.value.register_value_modifier(
            get_calibration_constant_pipeline_name(
                "maternal_disorders.transition_proportion"
            ),
            modifier=lambda: self.maternal_disorders_paf_data,
        )

        builder.value.register_attribute_modifier(
            "maternal_hemorrhage.transition_proportion",
            self.adjust_maternal_hemorrhage_proportion,
            required_resources=["hemoglobin.exposure", self.hemorrhage_relative_risk],
        )
        builder.value.register_value_modifier(
            get_calibration_constant_pipeline_name(
                "maternal_hemorrhage.transition_proportion"
            ),
            modifier=lambda: self.maternal_hemorrhage_paf_data,
        )

        builder.value.register_attribute_modifier(
            "hemoglobin.exposure",
            self.adjust_hemoglobin_exposure,
            required_resources=["maternal_hemorrhage"],
        )

        builder.population.register_initializer(
            self.initialize_hemoglobin,
            columns=[
                "hemoglobin_distribution_propensity",
                "hemoglobin_percentile",
                "hemoglobin_scale_factor",
            ],
            required_resources=[self.randomness],
        )

    def initialize_hemoglobin(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                "hemoglobin_distribution_propensity": self.randomness.get_draw(
                    pop_data.index, additional_key="hemoglobin_distribution_propensity"
                ),
                "hemoglobin_percentile": self.randomness.get_draw(
                    pop_data.index, additional_key="hemoglobin_percentile"
                ),
                "hemoglobin_scale_factor": self.randomness.choice(
                    pop_data.index,
                    choices=[
                        HEMOGLOBIN_SCALE_FACTOR_MODERATE_HEMORRHAGE,
                        HEMOGLOBIN_SCALE_FACTOR_SEVERE_HEMORRHAGE,
                    ],
                    p=[self.moderate_hemorrhage_probability, RESIDUAL_CHOICE],
                    additional_key="hemorrhage_scale_factors",
                ),
            },
            index=pop_data.index,
        )
        self.population_view.initialize(pop_update)

    def hemoglobin_source(self, idx: pd.Index) -> pd.Series:
        pop = self.population_view.get(
            idx,
            [
                "hemoglobin_distribution_propensity",
                "hemoglobin_percentile",
                "hemoglobin.exposure_parameters",
            ],
        )
        return self.sample_from_hemoglobin_distribution(
            pop["hemoglobin_distribution_propensity"],
            pop["hemoglobin_percentile"],
            pop["hemoglobin.exposure_parameters"],
        )

    @staticmethod
    def _gamma_ppf(propensity, mean, sd):
        """Returns the quantile for the given quantile rank (`propensity`) of a Gamma
        distribution with the specified mean and standard deviation.
        """
        shape = (mean / sd) ** 2
        scale = sd**2 / mean
        return scipy.stats.gamma(a=shape, scale=scale).ppf(propensity)

    @staticmethod
    def _mirrored_gumbel_ppf_2017(propensity, mean, sd):
        """Returns the quantile for the given quantile rank (`propensity`) of a mirrored Gumbel
        distribution with the specified mean and standard deviation.
        """
        x_max = HEMOGLOBIN_DISTRIBUTION_PARAMETERS.XMAX
        alpha = x_max - mean - (sd * np.euler_gamma * np.sqrt(6) / np.pi)
        scale = sd * np.sqrt(6) / np.pi
        return x_max - scipy.stats.gumbel_r(alpha, scale=scale).ppf(1 - propensity)

    def sample_from_hemoglobin_distribution(
        self, propensity_distribution, propensity, exposure_parameters
    ):
        """
        Returns a sample from an ensemble distribution with the specified mean and
        standard deviation (stored in `exposure_parameters`) that is 40% Gamma and
        60% mirrored Gumbel. The sampled value is a function of the two propensities
        `prop_dist` (used to choose whether to sample from the Gamma distribution or
        the mirrored Gumbel distribution) and `propensity` (used as the quantile rank
        for the selected distribution).
        """

        exposure_data = exposure_parameters
        mean = exposure_data["mean"]
        sd = exposure_data["stddev"]

        gamma = (
            propensity_distribution
            < HEMOGLOBIN_DISTRIBUTION_PARAMETERS.GAMMA_DISTRIBUTION_WEIGHT
        )
        gumbel = ~gamma

        ret_val = pd.Series(index=propensity_distribution.index, name="value", dtype=float)
        ret_val.loc[gamma] = self._gamma_ppf(propensity, mean, sd)[gamma]
        ret_val.loc[gumbel] = self._mirrored_gumbel_ppf_2017(propensity, mean, sd)[gumbel]
        return ret_val

    def adjust_maternal_disorder_proportion(self, index: pd.Index) -> pd.Series:
        hemoglobin_level = self.population_view.get(index, "hemoglobin.exposure")
        rr = self.maternal_disorders_relative_risk(index)
        ## annoyingly formatted
        tmrel = TMREL_HEMOGLOBIN_ON_MATERNAL_DISORDERS
        per_simulant_exposure = (
            (tmrel - hemoglobin_level + abs(tmrel - hemoglobin_level)) / 2 / RR_SCALAR
        )
        per_simulant_rr = rr**per_simulant_exposure
        return per_simulant_rr

    def adjust_maternal_hemorrhage_proportion(self, index: pd.Index) -> pd.Series:
        hemoglobin = self.population_view.get(index, "hemoglobin.exposure")
        threshold = SEVERE_ANEMIA_AMONG_PREGNANT_WOMEN_THRESHOLD
        severe_anemia_index = hemoglobin[hemoglobin < threshold].index

        rr = pd.Series(1.0, index=index)
        rr.loc[severe_anemia_index] = self.hemorrhage_relative_risk(severe_anemia_index)
        return rr

    def adjust_hemoglobin_exposure(
        self, index: pd.Index, hemoglobin_exposure: pd.DataFrame
    ) -> pd.DataFrame:
        # We need to persist this value for both current and recovered maternal hemorrhage
        # We don't need to undo after postpartum, as simulants become untracked
        hemoglobin_scale_factor = self.population_view.get(
            index,
            "hemoglobin_scale_factor",
            query="is_alive == True & maternal_hemorrhage != 'susceptible_to_maternal_hemorrhage'",
        )
        hemoglobin_exposure.loc[hemoglobin_scale_factor.index] *= hemoglobin_scale_factor
        return hemoglobin_exposure


class Anemia(Component):
    @property
    def time_step_priority(self) -> int:
        return 4

    def setup(self, builder: Builder):

        self.anemia_thresholds = self.build_lookup_table(
            builder,
            "anemia_thresholds",
            data_source=ANEMIA_THRESHOLD_DATA,
            value_columns=["severe", "moderate", "mild"],
        )
        builder.value.register_attribute_producer(
            "anemia_levels",
            source=self.anemia_source,
            required_resources=["hemoglobin.exposure", self.anemia_thresholds],
        )

        builder.value.register_attribute_producer(
            "anemia.disability_weight",
            source=self.compute_disability_weight,
            required_resources=["is_alive", "pregnancy"],
        )

        builder.value.register_attribute_modifier(
            "all_causes.disability_weight", modifier="anemia.disability_weight"
        )

        builder.population.register_initializer(
            self.initialize_anemia_status_at_birth, "anemia_status_at_birth"
        )

    def anemia_source(self, index: pd.Index) -> pd.Series:
        hemoglobin_level = self.population_view.get(index, "hemoglobin.exposure")
        thresholds = self.anemia_thresholds(index)

        choice_index = (hemoglobin_level.values[np.newaxis].T < thresholds).sum(axis=1)

        return pd.Series(
            np.array(["not_anemic", "mild", "moderate", "severe"])[choice_index],
            index=index,
            name="anemia_levels",
        )

    def compute_disability_weight(self, index: pd.Index):
        anemia_levels = self.population_view.get(index, "anemia_levels")
        raw_anemia_disability_weight = anemia_levels.map(ANEMIA_DISABILITY_WEIGHTS)
        dw_map = {
            models.NOT_PREGNANT_STATE_NAME: raw_anemia_disability_weight,
            models.PREGNANT_STATE_NAME: raw_anemia_disability_weight,
            ## Pause YLD accumulation during the parturition state
            models.PARTURITION_STATE_NAME: pd.Series(0, index=index),
            models.POSTPARTUM_STATE_NAME: raw_anemia_disability_weight,
        }
        pop = self.population_view.get(index, ["pregnancy", "is_alive"])
        disability_weight = pd.Series(np.nan, index=index)
        for state, dw in dw_map.items():
            in_state = (pop["is_alive"] == True) & (pop["pregnancy"] == state)
            disability_weight[in_state] = dw.loc[in_state]

        return disability_weight

    def initialize_anemia_status_at_birth(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame({"anemia_status_at_birth": "invalid"}, index=pop_data.index)
        self.population_view.initialize(pop_update)

    def on_time_step(self, event: Event):
        anemia_at_birth = self.population_view.get(
            event.index,
            "anemia_levels",
            query=("is_alive == True & pregnancy == 'parturition'"),
        )
        self.population_view.update(
            "anemia_status_at_birth",
            lambda _: anemia_at_birth.rename("anemia_status_at_birth"),
        )

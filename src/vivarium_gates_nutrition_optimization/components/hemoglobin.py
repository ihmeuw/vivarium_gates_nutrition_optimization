import numpy as np
import pandas as pd
import scipy.stats

from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

from vivarium_gates_nutrition_optimization.constants.data_values import (
    ANEMIA_DISABILITY_WEIGHTS,
    HEMOGLOBIN_DISTRIBUTION_PARAMETERS,
    HEMOGLOBIN_THRESHOLD_DATA,
    RR_SCALAR,
    SEVERE_ANEMIA_AMONG_PREGNANT_WOMEN_THRESHOLD,
    TMREL_HEMOGLOBIN_ON_MATERNAL_DISORDERS,
)

from vivarium_gates_nutrition_optimization.constants import data_keys


class Hemoglobin:
    """
    class for hemoglobin utilities and calculations that in turn will
    be used to find anemia status for simulants.
    """

    def __init__(self):
        pass

    @property
    def name(self):
        return "hemoglobin"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_created = [
            "hemoglobin_distribution_propensity",
            "hemoglobin_percentile",
        ]
        # load data
        mean = builder.data.load(data_keys.HEMOGLOBIN.MEAN)
        stddev = builder.data.load(data_keys.HEMOGLOBIN.STANDARD_DEVIATION)

        index_columns = [
            "sex",
            "age_start",
            "age_end",
            "year_start",
            "year_end",
        ]
        mean = mean.set_index(index_columns)["value"].rename("mean")
        stddev = stddev.set_index(index_columns)["value"].rename("stddev")
        distribution_parameters = pd.concat([mean, stddev], axis=1).reset_index()
        self.distribution_parameters = builder.value.register_value_producer(
            "hemoglobin.exposure_parameters",
            source=builder.lookup.build_table(
                distribution_parameters,
                key_columns=["sex"],
                parameter_columns=["age", "year"],
            ),
            requires_columns=["age", "sex"],
        )

        # Fix resource dependency cycle
        self.raw_hemoglobin = builder.value.register_value_producer(
            "raw_hemoglobin.exposure",
            source=self.hemoglobin_source,
            requires_values=["hemoglobin.exposure_parameters"],
            requires_streams=[self.name],
        )

        self.hemoglobin = builder.value.register_value_producer(
            "hemoglobin.exposure",
            source=self.raw_hemoglobin,
        )

        self.thresholds = builder.lookup.build_table(
            HEMOGLOBIN_THRESHOLD_DATA,
            key_columns=["sex", "pregnancy_status"],
            parameter_columns=["age"],
        )

        self.anemia_levels = builder.value.register_value_producer(
            "anemia_levels",
            source=self.anemia_source,
            requires_values=["hemoglobin.exposure"],
        )

        # self.maternal_disorder_risk_effect = builder.value.register_value_producer(
        #     "maternal_disorder_risk_effect",
        #     source=self.maternal_disorder_risk_effect,
        #     requires_values=["hemoglobin.exposure"],
        # )

        builder.value.register_value_producer(
            "anemia.disability_weight",
            source=self.disability_weight,
            requires_values=["hemoglobin.exposure"],
        )

        # self.hemorrhage_rr = builder.lookup.build_table(
        #         builder.data.load(data_keys.MATERNAL_DISORDERS.RR_MATERNAL_HEMORRHAGE_ATTRIBUTABLE_TO_HEMOGLOBIN),
        #         key_columns=["sex"],
        #         parameter_columns=["age", "year"],
        #     )

        # self.hemorrhage_paf = builder.lookup.build_table(
        #         builder.data.load(data_keys.MATERNAL_DISORDERS.PAF_MATERNAL_HEMORRHAGE_ATTRIBUTABLE_TO_HEMOGLOBIN),
        #         key_columns=["sex"],
        #         parameter_columns=["age", "year"],
        #     )

        # self.maternal_disorder_rr = builder.lookup.build_table(
        #     builder.data.load(data_keys.MATERNAL_DISORDERS.RR_MATERNAL_DISORDER_ATTRIBUTABLE_TO_HEMOGLOBIN),
        #     key_columns=["sex"],
        #     parameter_columns=["age", "year"],
        # )

        # self.maternal_disorder_paf = builder.lookup.build_table(
        #     builder.data.load(data_keys.MATERNAL_DISORDERS.PAF_MATERNAL_DISORDER_ATTRIBUTABLE_TO_HEMOGLOBIN),
        #     key_columns=["sex"],
        #     parameter_columns=["age", "year"],
        # )

        # builder.value.register_value_modifier(
        #     "probability_maternal_hemorrhage",
        #     self.adjust_maternal_hemorrhage_probability,
        #     requires_values=["hemoglobin.exposure"]
        # )

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.columns_created,
            requires_streams=[self.name],
        )

        self.population_view = builder.population.get_view(self.columns_created)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                "hemoglobin_distribution_propensity": self.randomness.get_draw(
                    pop_data.index, additional_key="hemoglobin_distribution_propensity"
                ),
                "hemoglobin_percentile": self.randomness.get_draw(
                    pop_data.index, additional_key="hemoglobin_percentile"
                ),
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def hemoglobin_source(self, idx: pd.Index) -> pd.Series:
        pop = self.population_view.get(idx)
        distribution_parameters = self.distribution_parameters(pop.index)
        return self.sample_from_hemoglobin_distribution(
            pop["hemoglobin_distribution_propensity"],
            pop["hemoglobin_percentile"],
            distribution_parameters,
        )

    def anemia_source(self, index: pd.Index) -> pd.Series:
        hemoglobin_level = self.hemoglobin(index)
        thresholds = self.thresholds(index)
        choice_index = (hemoglobin_level.values[np.newaxis].T < thresholds).sum(axis=1)

        return pd.Series(
            np.array(["none", "mild", "moderate", "severe"])[choice_index],
            index=index,
            name="anemia_levels",
        )

    # def maternal_disorder_risk_effect(self, index: pd.Index) -> pd.Series:
    #     hemoglobin_level = self.hemoglobin(index)
    #     rr = self.maternal_disorder_rr(index)
    #     paf = self.maternal_disorder_paf(index)["value"]
    #     tmrel = TMREL_HEMOGLOBIN_ON_MATERNAL_DISORDERS
    #     per_simulant_exposure = (tmrel - hemoglobin_level + abs(tmrel - hemoglobin_level)) / 2 / RR_SCALAR
    #     per_simulant_rr = rr ** per_simulant_exposure
    #     return (1 - paf) * per_simulant_rr

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

    def disability_weight(self, index: pd.Index) -> pd.Series:
        hemoglobin_level = self.hemoglobin(index)
        thresholds = self.thresholds(index)
        choice_index = (hemoglobin_level.values[np.newaxis].T < thresholds).sum(axis=1)
        anemia_levels = pd.Series(
            np.array(["none", "mild", "moderate", "severe"])[choice_index],
            index=index,
            name="anemia_levels",
        )
        return anemia_levels.map(ANEMIA_DISABILITY_WEIGHTS)

    # def adjust_maternal_hemorrhage_probability(self, index, probability):
    #     paf = self.hemorrhage_paf(index)["value"]
    #     rr = self.hemorrhage_rr(index)["value"]
    #     p_maternal_hemorrhage = probability["moderate_maternal_hemorrhage"] + probability["severe_maternal_hemorrhage"]
    #     severe_ratio = probability["severe_maternal_hemorrhage"] / p_maternal_hemorrhage
    #     p_maternal_hemorrhage_nonanemic = p_maternal_hemorrhage * (1 - paf)
    #     p_maternal_hemorrhage_anemic = p_maternal_hemorrhage_nonanemic * rr
    #     hemoglobin = self.hemoglobin(index)
    #     anemic = hemoglobin <= SEVERE_ANEMIA_AMONG_PREGNANT_WOMEN_THRESHOLD
    #     probability["severe_maternal_hemorrhage"] = severe_ratio * p_maternal_hemorrhage_nonanemic
    #     probability["moderate_maternal_hemorrhage"] = (1 - severe_ratio) * p_maternal_hemorrhage_nonanemic
    #     probability.loc[anemic, "severe_maternal_hemorrhage"] = severe_ratio * p_maternal_hemorrhage_anemic
    #     probability.loc[anemic, "moderate_maternal_hemorrhage"] = (1 - severe_ratio) * p_maternal_hemorrhage_anemic
    #     probability["not_maternal_hemorrhage"] = (1 - probability["moderate_maternal_hemorrhage"] - probability["severe_maternal_hemorrhage"])
    #     return probability

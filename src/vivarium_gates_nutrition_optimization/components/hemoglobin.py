import numpy as np
import pandas as pd
import scipy.stats
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RESIDUAL_CHOICE

from vivarium_gates_nutrition_optimization.constants import (
    data_keys,
    data_values,
    models,
)
from vivarium_gates_nutrition_optimization.constants.data_values import (
    RR_SCALAR,
    HEMOGLOBIN_SCALE_FACTOR_MODERATE_HEMORRHAGE,
    HEMOGLOBIN_SCALE_FACTOR_SEVERE_HEMORRHAGE,
    SEVERE_ANEMIA_AMONG_PREGNANT_WOMEN_THRESHOLD,
    TMREL_HEMOGLOBIN_ON_MATERNAL_DISORDERS,
    ANEMIA_DISABILITY_WEIGHTS,
    ANEMIA_THRESHOLD_DATA,
    HEMOGLOBIN_DISTRIBUTION_PARAMETERS,
)


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
        distribution_parameters = pd.concat([mean, stddev], axis=1).reset_index()

        self.hemorrhage_rr = builder.lookup.build_table(
            builder.data.load(data_keys.MATERNAL_DISORDERS.RR_ATTRIBUTABLE_TO_HEMOGLOBIN),
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )

        self.hemorrhage_paf = builder.lookup.build_table(
            builder.data.load(data_keys.MATERNAL_DISORDERS.PAF_ATTRIBUTABLE_TO_HEMOGLOBIN),
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )

        self.maternal_disorder_rr = builder.lookup.build_table(
            builder.data.load(data_keys.MATERNAL_DISORDERS.RR_ATTRIBUTABLE_TO_HEMOGLOBIN),
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )
        self.moderate_hemorrhage_probability = builder.data.load(
            data_keys.MATERNAL_HEMORRHAGE.MODERATE_HEMORRHAGE_PROBABILITY
        ).value.values[0]

        self.distribution_parameters = builder.value.register_value_producer(
            "hemoglobin.exposure_parameters",
            source=builder.lookup.build_table(
                distribution_parameters,
                key_columns=["sex"],
                parameter_columns=["age", "year"],
            ),
            requires_columns=["age", "sex"],
        )

        self.hemoglobin = builder.value.register_value_producer(
            "hemoglobin.exposure",
            source=self.hemoglobin_source,
            requires_values=["hemoglobin.exposure_parameters"],
            requires_streams=[self.name],
        )

        self.maternal_disorder_paf = builder.lookup.build_table(
            builder.data.load(data_keys.MATERNAL_DISORDERS.PAF_ATTRIBUTABLE_TO_HEMOGLOBIN),
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )
        builder.value.register_value_modifier(
            ## Hemoglobin affects MD
            "maternal_disorders.transition_probability",
            self.adjust_maternal_disorder_probability,
            requires_values=["hemoglobin.exposure"],
        )
        builder.value.register_value_modifier(
            ## Hemoglobin affcts MH
            "maternal_hemorrhage.transition_probability",
            self.adjust_maternal_hemorrhage_probability,
            requires_values=["hemoglobin.exposure"],
        )

        builder.value.register_value_modifier(
            ## MH affects hemoglobin
            "hemoglobin.exposure",
            self.adjust_hemoglobin_exposure,
            requires_columns=["maternal_hemorrhage"],
        )

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.columns_created,
            requires_streams=[self.name],
        )

        self.population_view = builder.population.get_view(
            self.columns_created + ["alive", "maternal_hemorrhage"]
        )

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

    def adjust_maternal_disorder_probability(
        self, index: pd.Index, maternal_disorder_probability: pd.DataFrame
    ) -> pd.Series:
        hemoglobin_level = self.hemoglobin(index)
        rr = self.maternal_disorder_rr(index)
        paf = self.maternal_disorder_paf(index)["value"]
        tmrel = TMREL_HEMOGLOBIN_ON_MATERNAL_DISORDERS
        per_simulant_exposure = (
            (tmrel - hemoglobin_level + abs(tmrel - hemoglobin_level)) / 2 / RR_SCALAR
        )
        per_simulant_rr = rr**per_simulant_exposure
        maternal_disorder_probability *= (1 - paf) * per_simulant_rr
        return maternal_disorder_probability.applymap(lambda value: 1 if value > 1 else value)

    def adjust_maternal_hemorrhage_probability(self, index, maternal_hemorrhage_probability):
        paf = self.hemorrhage_paf(index)["value"]
        rr = self.hemorrhage_rr(index)["value"]
        hemoglobin = self.hemoglobin(index)
        maternal_hemorrhage_probability *= 1 - paf
        # Dichotomous risk based on severe anemia
        maternal_hemorrhage_probability.loc[
            hemoglobin <= SEVERE_ANEMIA_AMONG_PREGNANT_WOMEN_THRESHOLD
        ] *= rr
        return maternal_hemorrhage_probability

    def adjust_hemoglobin_exposure(
        self, index: pd.Index, hemoglobin_exposure: pd.DataFrame
    ) -> pd.DataFrame:
        pop = self.population_view.get(index)
        maternal_hemorrhage_mask = (pop["alive"] == "alive") & (
            pop["maternal_hemorrhage"] == "maternal_hemorrhage"
        )
        if not maternal_hemorrhage_mask.any():
            return hemoglobin_exposure
        # Choose hemorrhage severity and multiply hemoglobin exposure by appropriate hemorrhage scale factor
        hemoglobin_exposure[maternal_hemorrhage_mask] *= self.randomness.choice(
            pop[maternal_hemorrhage_mask].index,
            choices=[
                HEMOGLOBIN_SCALE_FACTOR_MODERATE_HEMORRHAGE,
                HEMOGLOBIN_SCALE_FACTOR_SEVERE_HEMORRHAGE,
            ],
            p=[self.moderate_hemorrhage_probability, RESIDUAL_CHOICE],
            additional_key="hemorrhage_scale_factors",
        )
        return hemoglobin_exposure


class Anemia:
    @property
    def name(self):
        return "anemia"

    def setup(self, builder: Builder):
        self.hemoglobin = builder.value.get_value("hemoglobin.exposure")

        self.thresholds = builder.lookup.build_table(
            ANEMIA_THRESHOLD_DATA,
            key_columns=["sex", "pregnancy"],
            parameter_columns=["age"],
        )
        self.anemia_levels = builder.value.register_value_producer(
            "anemia_levels",
            source=self.anemia_source,
            requires_values=["hemoglobin.exposure"],
        )

        self.disability_weight = builder.value.register_value_producer(
            "anemia.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=["alive", "pregnancy"],
        )

        builder.value.register_value_modifier(
            "disability_weight",
            self.disability_weight,
        )

        self.population_view = builder.population.get_view(["alive", "pregnancy"])

    def anemia_source(self, index: pd.Index) -> pd.Series:
        hemoglobin_level = self.hemoglobin(index)
        thresholds = self.thresholds(index)
        choice_index = (hemoglobin_level.values[np.newaxis].T < thresholds).sum(axis=1)

        return pd.Series(
            np.array(["none", "mild", "moderate", "severe"])[choice_index],
            index=index,
            name="anemia_levels",
        )

    def compute_disability_weight(self, index: pd.Index):
        anemia_levels = self.anemia_levels(index)
        raw_anemia_disability_weight = anemia_levels.map(ANEMIA_DISABILITY_WEIGHTS)
        ## rescale postpartum YLDs the same way as in IV Iron
        postpartum_scalar = (
            data_values.DURATIONS.POSTPARTUM + data_values.DURATIONS.PARTURITION
        ) / data_values.DURATIONS.POSTPARTUM
        dw_map = {
            models.NOT_PREGNANT_STATE_NAME: raw_anemia_disability_weight,
            models.PREGNANT_STATE_NAME: raw_anemia_disability_weight,
            ## Pause YLD accumulation during the parturition state
            models.PARTURITION_STATE_NAME: pd.Series(0, index=index),
            models.POSTPARTUM_STATE_NAME: postpartum_scalar * raw_anemia_disability_weight,
        }

        pop = self.population_view.get(index)
        alive = pop["alive"] == "alive"
        disability_weight = pd.Series(np.nan, index=index)
        for state, dw in dw_map.items():
            in_state = alive & (pop["pregnancy"] == state)
            disability_weight[in_state] = dw.loc[in_state]

        return disability_weight

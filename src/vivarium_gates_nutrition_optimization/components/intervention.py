from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import scipy.stats
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RESIDUAL_CHOICE

from vivarium_gates_nutrition_optimization.constants import (
    data_keys,
    data_values,
    models,
)


class MaternalInterventions:
    configuration_defaults = {
        "intervention": {
            "scenario": "baseline",
        }
    }

    @property
    def name(self) -> str:
        return "maternal_interventions"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        start_date = builder.configuration.time.start
        self.start_date = datetime(start_date.year, start_date.month, start_date.day)
        self.randomness = builder.randomness.get_stream(self.name)

        self.scenario = builder.configuration.intervention.scenario
        self.ifa_coverage = builder.data.load(
            data_keys.MATERNAL_INTERVENTIONS.IFA_COVERAGE
        ).value[0]

        builder.value.register_value_modifier(
            "hemoglobin.exposure",
            self.update_exposure,
            requires_columns=[
                "intervention",
                "hemoglobin_effect_size",
            ],
        )

        self.columns_required = ["maternal_bmi_anemia_category"]
        self.columns_created = [
            "intervention",
            "hemoglobin_effect_size",
        ]
        self.population_view = builder.population.get_view(
            self.columns_required + self.columns_created + ["tracked"]
        )
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.columns_created,
            requires_columns=self.columns_required,
            requires_streams=[self.name],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop = self.population_view.subview(["maternal_bmi_anemia_category"]).get(
            pop_data.index
        )
        pop_update = pd.DataFrame(
            {"intervention": models.INVALID_TREATMENT, "hemoglobin_effect_size": np.nan},
            index=pop.index,
        )
        low_bmi = pop["maternal_bmi_anemia_category"].isin(
            [models.LOW_BMI_NON_ANEMIC, models.LOW_BMI_ANEMIC]
        )
        coverage = data_values.INTERVENTION_SCENARIO_COVERAGE.loc[self.scenario]
        pop_update["intervention"] = np.where(
            low_bmi, coverage["low_bmi"], coverage["normal_bmi"]
        )

        unsampled_ifa = pop_update["intervention"] == "maybe_ifa"
        pop_update.loc[unsampled_ifa, "intervention"] = self.randomness.choice(
            pop_update[unsampled_ifa].index,
            choices=[models.IFA_SUPPLEMENTATION, models.NO_TREATMENT],
            p=[self.ifa_coverage, RESIDUAL_CHOICE],
            additional_key="ifa_coverage",
        )

        hemoglobin_shift_propensity = self.randomness.get_draw(
            pop_data.index, "hemoglobin_shift_propensity"
        )
        loc, scale = data_values.IFA_EFFECT_SIZE
        pop_update["hemoglobin_effect_size"] = scipy.stats.norm.ppf(
            hemoglobin_shift_propensity, loc=loc, scale=scale
        )

        self.population_view.update(pop_update)

    def update_exposure(self, index, exposure):
        if self.clock() - self.start_date >= timedelta(
            days=data_values.DURATIONS.INTERVENTION_DELAY
        ):
            pop = self.population_view.get(index)
            on_treatment = pop["intervention"] != models.NO_TREATMENT
            exposure.loc[on_treatment] += pop.loc[on_treatment, "hemoglobin_effect_size"]

        return exposure

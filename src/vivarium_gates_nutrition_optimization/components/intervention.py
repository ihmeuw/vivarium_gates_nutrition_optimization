from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RESIDUAL_CHOICE
from vivarium.framework.time import get_time_stamp

from vivarium_gates_nutrition_optimization.constants import (
    data_keys,
    data_values,
    models,
)


class MaternalInterventions(Component):
    CONFIGURATION_DEFAULTS = {
        "intervention": {
            "scenario": "baseline",
        }
    }

    @property
    def columns_created(self) -> List[str]:
        return ["intervention"]

    @property
    def columns_required(self) -> List[str]:
        return ["maternal_bmi_anemia_category", "tracked"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_streams": [self.name], "requires_columns": [self.columns_required]}

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.start_date = get_time_stamp(builder.configuration.time.start)
        self.randomness = builder.randomness.get_stream(self.name)

        self.scenario = builder.configuration.intervention.scenario
        self.ifa_coverage = builder.data.load(
            data_keys.MATERNAL_INTERVENTIONS.IFA_COVERAGE
        ).value[0]
        self.mms_stillbirth_rr = builder.data.load(
            data_keys.MATERNAL_INTERVENTIONS.MMS_STILLBIRTH_RR
        ).value[0]
        self.bep_stillbirth_rr = builder.data.load(
            data_keys.MATERNAL_INTERVENTIONS.BEP_STILLBIRTH_RR
        ).value[0]
        self.ifa_effect_size = builder.data.load(
            data_keys.MATERNAL_INTERVENTIONS.IFA_EFFECT_SIZE
        ).value[0]

        builder.value.register_value_modifier(
            "hemoglobin.exposure",
            self.update_exposure,
            requires_columns=self.columns_created,
        )

        builder.value.register_value_modifier(
            "birth_outcome_probabilities",
            self.adjust_stillbirth_probability,
            requires_columns=self.columns_created,
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop = self.population_view.subview(["maternal_bmi_anemia_category"]).get(
            pop_data.index
        )
        
        if self.scenario == 'ifa':
            pop_update = pd.DataFrame(
                {"intervention": 'ifa'},
                index=pop.index,
            )
        else:
            pop_update = pd.DataFrame(
                {"intervention": None},
                index=pop.index,
            )
            baseline_ifa = self.randomness.choice(
                pop.index,
                choices=[models.IFA_SUPPLEMENTATION, models.NO_TREATMENT],
                p=[self.ifa_coverage, RESIDUAL_CHOICE],
                additional_key="baseline_ifa",
            )
            low_bmi = pop["maternal_bmi_anemia_category"].isin(
                [models.LOW_BMI_NON_ANEMIC, models.LOW_BMI_ANEMIC]
            )
            coverage = data_values.INTERVENTION_SCENARIO_COVERAGE.loc[self.scenario]
            pop_update["intervention"] = np.where(
                low_bmi, coverage["low_bmi"], coverage["normal_bmi"]
            )

            unsampled_ifa = pop_update["intervention"] == "maybe_ifa"
            pop_update.loc[unsampled_ifa, "intervention"] = baseline_ifa.loc[unsampled_ifa]

        self.population_view.update(pop_update)

    def update_exposure(self, index, exposure):
        if self.clock() - self.start_date >= timedelta(
            days=data_values.DURATIONS.INTERVENTION_DELAY_DAYS
        ):
            pop = self.population_view.get(index)
            exposure.loc[pop["intervention"] == models.NO_TREATMENT] -= (
                self.ifa_coverage * self.ifa_effect_size
            )
            exposure.loc[pop["intervention"] != models.NO_TREATMENT] += (
                1 - self.ifa_coverage
            ) * self.ifa_effect_size

        return exposure

    def adjust_stillbirth_probability(self, index, birth_outcome_probabilities):
        pop = self.population_view.subview(["intervention"]).get(index)
        rrs = {
            models.MMS_SUPPLEMENTATION: self.mms_stillbirth_rr,
            models.BEP_SUPPLEMENTATION: self.bep_stillbirth_rr,
        }
        for intervention, rr in rrs.items():
            on_treatment = pop["intervention"] == intervention
            # Add spare probability onto live births first
            birth_outcome_probabilities.loc[
                on_treatment, models.LIVE_BIRTH_OUTCOME
            ] += birth_outcome_probabilities.loc[on_treatment, models.STILLBIRTH_OUTCOME] * (
                1 - rr
            )
            # Then re-scale stillbirth probability
            birth_outcome_probabilities.loc[on_treatment, models.STILLBIRTH_OUTCOME] *= rr
            # This preserves normalization by construction

        return birth_outcome_probabilities

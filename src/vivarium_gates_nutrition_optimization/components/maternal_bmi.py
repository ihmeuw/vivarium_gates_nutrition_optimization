from typing import Dict, List

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_nutrition_optimization.constants import (
    data_keys,
    data_values,
    models,
)


class MaternalBMIExposure(Component):
    @property
    def columns_created(self) -> List[str]:
        return ["maternal_bmi_propensity", "maternal_bmi_anemia_category"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_values": ["raw_hemoglobin.exposure"],
            "requires_streams": [self.name],
        }

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.hemoglobin = builder.value.get_value("raw_hemoglobin.exposure")
        self.threshold = data_values.MATERNAL_BMI_ANEMIA_THRESHOLD

        self.probability_low_given_anemic = builder.lookup.build_table(
            builder.data.load(data_keys.MATERNAL_BMI.PREVALENCE_LOW_BMI_ANEMIC),
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )
        self.probability_low_given_non_anemic = builder.lookup.build_table(
            builder.data.load(data_keys.MATERNAL_BMI.PREVALENCE_LOW_BMI_NON_ANEMIC),
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )

    def on_initialize_simulants(self, pop_data: SimulantData):
        index = pop_data.index
        propensity = self.randomness.get_draw(index)
        p_low_anemic = self.probability_low_given_anemic(index)
        p_low_non_anemic = self.probability_low_given_non_anemic(index)
        hemoglobin = self.hemoglobin(index)
        anemic = index[hemoglobin < self.threshold]
        non_anemic = index.difference(anemic)

        bmi = pd.Series(models.INVALID_BMI_ANEMIA, index=index)

        bmi[anemic] = np.where(
            propensity.loc[anemic] < p_low_anemic.loc[anemic],
            models.LOW_BMI_ANEMIC,
            models.NORMAL_BMI_ANEMIC,
        )
        bmi[non_anemic] = np.where(
            propensity.loc[non_anemic] < p_low_non_anemic.loc[non_anemic],
            models.LOW_BMI_NON_ANEMIC,
            models.NORMAL_BMI_NON_ANEMIC,
        )

        pop_update = pd.concat(
            [
                propensity.rename("maternal_bmi_propensity"),
                bmi.rename("maternal_bmi_anemia_category"),
            ],
            axis=1,
        )

        self.population_view.update(pop_update)

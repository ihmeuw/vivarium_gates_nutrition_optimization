from typing import Dict

import numpy as np
import pandas as pd
import scipy.stats
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_nutrition_optimization.constants import (
    data_keys,
    data_values,
    models,
)


class MaternalInterventions:
    configuration_defaults = {
        'intervention': {
            'start_year': 2025,
            'scenario': 'baseline',
        }
    }

    @property
    def name(self) -> str:
        return 'maternal_interventions'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.randomness = builder.randomness.get_stream(self.name)
        self.coverage = self._load_intervention_coverage(builder)
        self.hemoglobin = builder.value.get_value('hemoglobin.exposure')

        self.effect_sizes = pd.DataFrame()

        builder.value.register_value_modifier(
            'hemoglobin.exposure',
            self.update_exposure,
            requires_columns=[
                'intervention',
                'hemoglobin_effect_size',
            ],
        )

        self.columns_required = [
            'maternal_bmi_anemia_category'
        ]
        self.columns_created = [
            'intervention',
            'hemoglobin_effect_size',

        ]
        self.population_view = builder.population.get_view(
            self.columns_required + self.columns_created
        )
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.columns_created,
            requires_columns=self.columns_required,
            requires_streams=[self.name],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        treatment_propensity = self.randomness.get_draw(pop_data.index, 'treatment_propensity')
        anemic = self.hemoglobin(pop_data.index) < data_values.IV_IRON_THRESHOLD



        hemoglobin_shift_propensity = self.randomness.get_draw(pop_data.index, 'hemoglobin_shift_propensity')
        loc, scale = data_values.IFA_EFFECT_SIZE
        pop_update['hemoglobin_effect_size'] = scipy.stats.norm.ppf(hemoglobin_shift_propensity, loc=loc, scale=scale)
  
        self.population_view.update(pop_update)

    def update_exposure(self, index, exposure):
        time = 8 * 7
        if time > data_values.INTERVENTION_EFFECT_START:
            pop = self.population_view.get(index)
            on_treatment = pop['intervention'] != models.NO_TREATMENT
            exposure.loc[on_treatment] *=  pop[on_treatment, 'hemoglobin_effect_size']

        return exposure



    def _sample_oral_iron_status(
        self,
        propensity: pd.Series,
        coverage: pd.Series,
    ) -> pd.Series:
        index = propensity.index
        bmi_status = self.population_view.subview(['maternal_bmi_anemia_category']).get(index)
        underweight = (bmi_status['maternal_bmi_anemia_category']
                       .isin([models.LOW_BMI_ANEMIC, models.LOW_BMI_NON_ANEMIC]))

        coverage.loc[models.NO_TREATMENT] = 1 - coverage.sum()
        p_covered = pd.DataFrame(coverage.to_dict(), index=index).cumsum(axis=1)
        choice_index = (propensity.values[np.newaxis].T > p_covered).sum(axis=1)

        supplementation = pd.Series(p_covered.columns[choice_index], index=index)
        other = supplementation == 'other'
        supplementation.loc[other & underweight] = models.BEP_SUPPLEMENTATION
        supplementation.loc[other & ~underweight] = models.MMS_SUPPLEMENTATION
        return supplementation

    def _sample_baseline_oral_iron_status(
        self,
        propensity: pd.Series,
        coverage: pd.Series,
    ) -> pd.Series:
        return pd.Series(
            np.where(propensity < coverage, models.TREATMENT, models.NO_TREATMENT),
            index=propensity.index,
        )


    def _load_intervention_coverage(self, builder: Builder) -> pd.DataFrame:
        scenario = builder.configuration.intervention.scenario
        year_start = int(builder.configuration.intervention.start_year)

        data = builder.data.load(data_keys.MATERNAL_INTERVENTIONS.COVERAGE)
        data = data.set_index(['scenario', 'year', 'intervention']).unstack()
        data.columns = data.columns.droplevel()
        data = (data
                .reset_index()
                .drop(columns='mms')
                .rename(columns={'bep': 'other'}))

        data['time'] = (
            pd.Timestamp(f"{year_start}-1-1")
            + pd.to_timedelta(365.25 * (data['year'] - data['year'].min()), unit='D')
        )
        data = data.set_index(['scenario', 'time']).drop(columns='year')

        coverage = pd.concat([
            data.loc['baseline', 'ifa'].rename('baseline_ifa'),
            data.loc[scenario],
        ], axis=1)

        return coverage
from typing import Dict

import numpy as np
import pandas as pd
import scipy.stats
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_iv_iron.constants import (
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
                'maternal_supplementation',
                'antenatal_iv_iron',
                'postpartum_iv_iron',
            ],
        )

        self.columns_required = [
            'pregnancy_status',
            'pregnancy_state_change_date',
            'maternal_bmi_anemia_category'
        ]
        self.columns_created = [
            'treatment_propensity',
            'baseline_ifa',
            'baseline_ifa_date',
            'maternal_supplementation',
            'maternal_supplementation_date',
            'antenatal_iv_iron',
            'antenatal_iv_iron_date',
            'postpartum_iv_iron',
            'postpartum_iv_iron_date',

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
        builder.event.register_listener(
            'time_step',
            self.on_time_step,
            priority=8,  # After pregnancy state changes
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.effect_sizes = pd.concat([
            self.effect_sizes,
            self._sample_effect_sizes(pop_data.index),
        ])
        pregnant, postpartum, anemic, in_treatment_window = self._get_indicators(
            pop_data.index, pop_data.creation_time,
        )
        propensity = self.randomness.get_draw(pop_data.index).rename('treatment_propensity')
        sampling_map = {
            'maternal_supplementation': (
                (pregnant & in_treatment_window(7 * 8)) | postpartum,
                ['ifa', 'other'],
                self._sample_oral_iron_status
             ),
            'baseline_ifa': (
                (pregnant & in_treatment_window(7 * 8)) | postpartum,
                'ifa',
                self._sample_baseline_oral_iron_status
            ),
            'antenatal_iv_iron': (
                anemic & ((pregnant & in_treatment_window(7 * 15)) | postpartum),
                'antenatal_iv_iron',
                self._sample_iv_iron_status
            ),
            'postpartum_iv_iron': (
                anemic & postpartum,
                'postpartum_iv_iron',
                self._sample_iv_iron_status
            ),
        }
        pop_update = pd.DataFrame({
            'treatment_propensity': propensity,
            **{k: models.INVALID_TREATMENT for k in sampling_map},
            **{f"{k}_date": pd.NaT for k in sampling_map},
        }, index=pop_data.index)
        pop_update = self._sample_intervention_status(
            pop_update, pop_data.creation_time, sampling_map
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index).copy()
        pregnant, postpartum, anemic, in_treatment_window = self._get_indicators(
            event.index, event.time, event.step_size,
        )
        sampling_map = {
            'maternal_supplementation': (
                pregnant & in_treatment_window(7 * 8),
                ['ifa', 'other'],
                self._sample_oral_iron_status
            ),
            'baseline_ifa': (
                pregnant & in_treatment_window(7 * 8),
                'ifa',
                self._sample_baseline_oral_iron_status
            ),
            'antenatal_iv_iron': (
                anemic & pregnant & in_treatment_window(7 * 15),
                'antenatal_iv_iron',
                self._sample_iv_iron_status
            ),
            'postpartum_iv_iron': (
                anemic & postpartum & in_treatment_window(7),
                'postpartum_iv_iron',
                self._sample_iv_iron_status),
        }

        pop_update = self._sample_intervention_status(
            pop, event.time, sampling_map,
        )
        intervention_over = (
            (pop['pregnancy_status'] == models.NOT_PREGNANT_STATE)
            & (pop['pregnancy_state_change_date'] == event.time)
        )
        for intervention in ['baseline_ifa',
                             'maternal_supplementation',
                             'antenatal_iv_iron',
                             'postpartum_iv_iron']:
            pop_update.loc[intervention_over, intervention] = models.INVALID_TREATMENT
            pop_update.loc[intervention_over, f'{intervention}_date'] = pd.NaT

        self.population_view.update(pop_update)

    def update_exposure(self, index, exposure):
        pop = self.population_view.get(index)
        effect_sizes = self.effect_sizes.loc[pop.index]
        baseline_ifa_coverage = self._get_coverage(self.clock()).loc['baseline_ifa']

        has_ifa_status = (pop['baseline_ifa'] != models.INVALID_TREATMENT).rename(None)
        exposure.loc[has_ifa_status] -= (
            baseline_ifa_coverage
            * effect_sizes.loc[has_ifa_status, 'maternal_supplementation']
        )
        for treatment in effect_sizes:
            on_treatment = ~pop[treatment].isin(
                [models.INVALID_TREATMENT, models.NO_TREATMENT]
            ).rename(None)
            exposure.loc[on_treatment] += effect_sizes.loc[on_treatment, treatment]
        return exposure

    def _sample_intervention_status(
        self,
        pop_update: pd.DataFrame,
        time: pd.Timestamp,
        sampling_map: Dict
    ):
        index = pop_update.index
        coverage = self._get_coverage(time)
        for name, (eligibility_mask, coverage_columns, sampling_func) in sampling_map.items():
            eligible = index[eligibility_mask]
            pop_update.loc[eligible, name] = sampling_func(
                pop_update.loc[eligible, 'treatment_propensity'],
                coverage.loc[coverage_columns],
            )
            pop_update.loc[eligible, f'{name}_date'] = time
        return pop_update

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

    def _sample_iv_iron_status(
        self,
        propensity: pd.Series,
        coverage: float,
    ) -> pd.Series:
        return pd.Series(
            np.where(propensity < coverage, models.TREATMENT, models.NO_TREATMENT),
            index=propensity.index
        )

    def _get_indicators(
        self,
        index: pd.Index,
        time: pd.Timestamp,
        step_size: pd.Timedelta = None,
    ):
        cols = ['pregnancy_status', 'pregnancy_state_change_date']
        pop = self.population_view.subview(cols).get(index)
        pregnant = pop['pregnancy_status'] == models.PREGNANT_STATE
        postpartum = pop['pregnancy_status'].isin([
            models.MATERNAL_DISORDER_STATE,
            models.NO_MATERNAL_DISORDER_STATE,
            models.POSTPARTUM_STATE,
        ])
        days_since_event = (time - pop['pregnancy_state_change_date']).dt.days

        if step_size is not None:
            # On time step. Check time to treat is within the current step.
            hemoglobin = self.hemoglobin(index)
            anemic = hemoglobin < data_values.IV_IRON_THRESHOLD

            days_to_next_event = (
                time + step_size - pop['pregnancy_state_change_date']
            ).dt.days

            def in_window(time_to_treat: int):
                return ((days_since_event <= time_to_treat)
                        & (time_to_treat < days_to_next_event))
        else:
            hemoglobin = self.hemoglobin.source(index)
            anemic = hemoglobin < data_values.IV_IRON_THRESHOLD
            # On initialize sims.  Check time to treat is in the past.
            def in_window(time_to_treat: int):
                return time_to_treat < days_since_event

        return pregnant, postpartum, anemic, in_window

    def _get_coverage(self, time: pd.Timestamp):
        if time < self.coverage.index.min():
            coverage = self.coverage.iloc[0]
        else:
            coverage = self.coverage.copy()
            coverage.loc[time, :] = np.nan
            coverage = coverage.sort_index().interpolate().loc[time]

        return coverage

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

    def _sample_effect_sizes(self, index: pd.Index) -> pd.DataFrame:
        loc, scale = data_values.IFA_EFFECT_SIZE
        supp_rvs = self.randomness.get_draw(index, 'maternal_supplementation_effect')
        maternal_supplementation = scipy.stats.norm.ppf(supp_rvs, loc=loc, scale=scale)

        loc, scale = data_values.IV_IRON_EFFECT_SIZE
        iv_rvs = self.randomness.get_draw(index, 'iv_iron_effect')
        iv_iron = scipy.stats.norm.ppf(iv_rvs, loc=loc, scale=scale)
        iv_iron[iv_iron < 0] = 0
        return pd.DataFrame({
            'maternal_supplementation': maternal_supplementation,
            'antenatal_iv_iron': iv_iron,
            'postpartum_iv_iron': iv_iron,
        }, index=index)


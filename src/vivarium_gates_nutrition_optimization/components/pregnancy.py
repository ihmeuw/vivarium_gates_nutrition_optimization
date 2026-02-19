from typing import Dict, List

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.values import Pipeline
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

from vivarium_gates_nutrition_optimization.components.children import NewChildren
from vivarium_gates_nutrition_optimization.constants import data_keys, models
from vivarium_gates_nutrition_optimization.constants.data_values import DURATIONS
from vivarium_gates_nutrition_optimization.constants.metadata import (
    ARTIFACT_INDEX_COLUMNS,
)


class NotPregnantState(SusceptibleState):
    def __init__(self, state_id, *args, **kwargs):
        super(SusceptibleState, self).__init__(state_id, *args, name_prefix="not_", **kwargs)


class PregnantState(DiseaseState):
    def __init__(self, state_id, *args, **kwargs):
        super().__init__(state_id, *args, **kwargs)
        self.new_children = NewChildren()

    ##############
    # Properties #
    ##############

    @property
    def sub_components(self):
        return super().sub_components + [self.new_children]

    def setup(self, builder: Builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super().setup(builder)
        self.time_step = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)

        self.birth_outcome_probabilities_lookup = self.build_lookup_table(
            builder,
            "birth_outcome_probabilities",
            data_source=get_birth_outcome_probabilities(builder),
            value_columns=["live_birth", "partial_term", "stillbirth"],
        )

        builder.value.register_attribute_producer(
            "birth_outcome_probabilities",
            source=self.birth_outcome_probabilities_lookup,
        )

        # NOTE: event times and event counts are already registered by the BaseDiseaseState
        builder.population.register_initializer(
            self.initialize_child_population,
            columns=[
                "pregnancy_outcome",
                "pregnancy_duration",
                "sex_of_child",
                "birth_weight",
                "gestational_age",
            ],
            required_resources=[self.model, self.randomness, "birth_outcome_probabilities"],
        )

    def initialize_child_population(self, pop_data: SimulantData) -> None:
        for transition in self.transition_set:
            if transition.start_active:
                transition.set_active(pop_data.index)

        pop_events = self.get_initial_event_times(pop_data)
        pregnancy_outcomes_and_durations = self.sample_pregnancy_outcomes_and_durations(
            pop_data
        )
        pop_update = pd.concat([pop_events, pregnancy_outcomes_and_durations], axis=1)
        self.population_view.update(pop_update)

    def sample_pregnancy_outcomes_and_durations(self, pop_data: SimulantData) -> pd.DataFrame:
        # Order the columns so that partial_term isn't in the middle!
        outcome_probabilities = self.population_view.get_attribute_frame(
            pop_data.index, "birth_outcome_probabilities"
        )
        pregnancy_outcomes = pd.DataFrame(
            {
                "pregnancy_outcome": self.randomness.choice(
                    pop_data.index,
                    choices=outcome_probabilities.columns.to_list(),
                    p=outcome_probabilities,
                    additional_key="pregnancy_outcome",
                )
            }
        )

        term_child_map = {
            models.STILLBIRTH_OUTCOME: self.sample_full_term_durations,
            models.LIVE_BIRTH_OUTCOME: self.sample_full_term_durations,
            models.PARTIAL_TERM_OUTCOME: self.sample_partial_term_durations,
        }

        for term_length, sampling_function in term_child_map.items():
            term_pop = pregnancy_outcomes[
                pregnancy_outcomes["pregnancy_outcome"] == term_length
            ].index
            pregnancy_outcomes.loc[
                term_pop,
                ["sex_of_child", "birth_weight", "gestational_age", "pregnancy_duration"],
            ] = sampling_function(term_pop)

        return pregnancy_outcomes

    def sample_partial_term_durations(self, partial_term_pop: pd.Index) -> pd.DataFrame:
        child_status = self.new_children.empty(partial_term_pop)
        low, high = DURATIONS.DETECTION_DAYS, DURATIONS.PARTIAL_TERM_DAYS
        draw = self.randomness.get_draw(
            partial_term_pop, additional_key="partial_term_pregnancy_duration"
        )
        child_status["pregnancy_duration"] = pd.to_timedelta(
            (low + (high - low) * draw), unit="days"
        )
        return child_status

    def sample_full_term_durations(self, full_term_pop: pd.Index) -> pd.DataFrame:
        child_status = self.new_children.generate_children(full_term_pop)
        child_status["pregnancy_duration"] = pd.to_timedelta(
            7 * child_status["gestational_age"], unit="days"
        )
        return child_status

    def get_initial_event_times(self, pop_data: SimulantData) -> pd.DataFrame:
        """Overwrite the BaseDiseaseState method"""
        return pd.DataFrame(
            {
                self.event_time_column: self.clock() + self.time_step(),
                self.event_count_column: 1,
            },
            index=pop_data.index,
        )


class PregnancyModel(DiseaseModel):
    def __init__(self, cause: str, **kwargs):
        super().__init__(cause, **kwargs)
        self._name = f"disease_model.{cause}"

    @property
    def time_step_priority(self) -> int:
        return 3


def Pregnancy():
    not_pregnant = NotPregnantState(models.PREGNANT_STATE_NAME)
    pregnant = PregnantState(
        models.PREGNANT_STATE_NAME,
        allow_self_transition=True,
        prevalence=1.0,
        dwell_time=0.0,
        disability_weight=0.0,
        excess_mortality_rate=0.0,
    )
    parturition = DiseaseState(
        models.PARTURITION_STATE_NAME,
        allow_self_transition=True,
        prevalence=0.0,
        dwell_time=lambda builder: builder.time.step_size()(),
        disability_weight=0.0,
        excess_mortality_rate=0.0,
    )
    postpartum = DiseaseState(
        models.POSTPARTUM_STATE_NAME,
        allow_self_transition=True,
        prevalence=0.0,
        dwell_time=lambda *_: pd.Timedelta(days=DURATIONS.POSTPARTUM_DAYS),
        disability_weight=0.0,
        excess_mortality_rate=0.0,
    )

    pregnant.add_dwell_time_transition(parturition)

    parturition.add_dwell_time_transition(postpartum)

    postpartum.add_dwell_time_transition(not_pregnant)

    return PregnancyModel(
        models.PREGNANCY_MODEL_NAME,
        states=[not_pregnant, pregnant, parturition, postpartum],
        cause_specific_mortality_rate=0.0,
    )


def get_birth_outcome_probabilities(builder: Builder) -> pd.DataFrame:
    asfr = builder.data.load(data_keys.PREGNANCY.ASFR).set_index(ARTIFACT_INDEX_COLUMNS)
    sbr = (
        builder.data.load(data_keys.PREGNANCY.SBR)
        .set_index("year_start")
        .drop(columns=["year_end"])
        .reindex(asfr.index, level="year_start")
    )

    raw_incidence_miscarriage = builder.data.load(
        data_keys.PREGNANCY.RAW_INCIDENCE_RATE_MISCARRIAGE
    ).set_index(ARTIFACT_INDEX_COLUMNS)
    raw_incidence_ectopic = builder.data.load(
        data_keys.PREGNANCY.RAW_INCIDENCE_RATE_ECTOPIC
    ).set_index(ARTIFACT_INDEX_COLUMNS)

    total_incidence = (
        asfr
        + asfr.multiply(sbr["value"], axis=0)
        + raw_incidence_ectopic
        + raw_incidence_miscarriage
    )

    partial_term = (raw_incidence_ectopic + raw_incidence_miscarriage) / total_incidence
    partial_term["pregnancy_outcome"] = models.PARTIAL_TERM_OUTCOME
    live_births = asfr / total_incidence
    live_births["pregnancy_outcome"] = models.LIVE_BIRTH_OUTCOME
    stillbirths = asfr.multiply(sbr["value"], axis=0) / total_incidence
    stillbirths["pregnancy_outcome"] = models.STILLBIRTH_OUTCOME
    probabilities = pd.concat([partial_term, live_births, stillbirths])
    probabilities = probabilities.pivot(
        columns="pregnancy_outcome", values="value"
    ).reset_index()
    return probabilities


class UntrackNotPregnant(Component):
    """Component for untracking not pregnant simulants"""

    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        builder.value.register_attribute_modifier("exit_time", self.update_exit_times)
        builder.population.register_tracked_query(
            f"pregnancy != '{models.NOT_PREGNANT_STATE_NAME}'"
        )

    def update_exit_times(self, index: pd.Index, target: pd.Series) -> None:
        """Update exit times for simulants who are no longer pregnant."""
        not_pregnant_idx = self.population_view.get_filtered_index(
            index,
            query=f"pregnancy == '{models.NOT_PREGNANT_STATE_NAME}'",
        )
        newly_not_pregnant_idx = not_pregnant_idx.intersection(target[target.isna()].index)
        target.loc[newly_not_pregnant_idx] = self.clock() + self.step_size()
        return target

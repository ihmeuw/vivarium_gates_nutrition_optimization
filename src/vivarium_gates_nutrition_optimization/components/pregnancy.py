from typing import Dict, List

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.utilities import is_non_zero

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
    def columns_created(self):
        return [
            self.event_time_column,
            self.event_count_column,
            "pregnancy_outcome",
            "pregnancy_duration",
            "sex_of_child",
            "birth_weight",
            "gestational_age",
        ]

    @property
    def sub_components(self):
        return super().sub_components + [self.new_children]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": [self.model],
            "requires_values": ["birth_outcome_probabilities"],
            "requires_streams": [],
        }

    def setup(self, builder: Builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super(DiseaseState, self).setup(builder)
        self.clock = builder.time.clock()

        prevalence_data = self.load_prevalence_data(builder)
        self.prevalence = builder.lookup.build_table(
            prevalence_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

        birth_prevalence_data = self.load_birth_prevalence_data(builder)
        self.birth_prevalence = builder.lookup.build_table(
            birth_prevalence_data, key_columns=["sex"], parameter_columns=["year"]
        )

        self.dwell_time = self.get_dwell_time_pipeline(builder)

        disability_weight_data = self.load_disability_weight_data(builder)
        self.has_disability = is_non_zero(disability_weight_data)
        self.base_disability_weight = builder.lookup.build_table(
            disability_weight_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.disability_weight = builder.value.register_value_producer(
            f"{self.state_id}.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=["age", "sex", "alive", self.model],
        )
        builder.value.register_value_modifier(
            "disability_weight", modifier=self.disability_weight
        )

        excess_mortality_data = self.load_excess_mortality_rate_data(builder)
        self.has_excess_mortality = is_non_zero(excess_mortality_data)
        self.base_excess_mortality_rate = builder.lookup.build_table(
            excess_mortality_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.excess_mortality_rate = builder.value.register_rate_producer(
            self.excess_mortality_rate_pipeline_name,
            source=self.compute_excess_mortality_rate,
            requires_columns=["age", "sex", "alive", self.model],
            requires_values=[self.excess_mortality_rate_paf_pipeline_name],
        )
        paf = builder.lookup.build_table(0)
        self.joint_paf = builder.value.register_value_producer(
            self.excess_mortality_rate_paf_pipeline_name,
            source=lambda idx: [paf(idx)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )
        builder.value.register_value_modifier(
            "mortality_rate",
            modifier=self.adjust_mortality_rate,
            requires_values=[self.excess_mortality_rate_pipeline_name],
        )

        self.randomness_prevalence = builder.randomness.get_stream(
            f"{self.state_id}_prevalent_cases"
        )
        self.time_step = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)

        self.birth_outcome_probabilities = builder.value.register_value_producer(
            "birth_outcome_probabilities",
            source=builder.lookup.build_table(
                get_birth_outcome_probabilities(builder),
                key_columns=["sex"],
                parameter_columns=["age", "year"],
            ),
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
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
        outcome_probabilities = self.birth_outcome_probabilities(pop_data.index)[
            ["partial_term", "stillbirth", "live_birth"]
        ]
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

    def get_dwell_time_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            f"{self.state_id}.dwell_time",
            source=lambda index: self.population_view.get(index)["pregnancy_duration"],
            requires_columns=["age", "sex", "pregnancy_outcome"],
        )

    def get_initial_event_times(self, pop_data: SimulantData) -> pd.DataFrame:
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
        get_data_functions={
            "prevalence": lambda *_: 1.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0,
            # Add a dummy dwell time so we can overwrite it later
            "dwell_time": lambda *_: None,
        },
    )
    parturition = DiseaseState(
        models.PARTURITION_STATE_NAME,
        allow_self_transition=True,
        get_data_functions={
            "prevalence": lambda *_: 0.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0,
            "dwell_time": lambda builder, cause: builder.time.step_size()(),
        },
    )
    postpartum = DiseaseState(
        models.POSTPARTUM_STATE_NAME,
        allow_self_transition=True,
        get_data_functions={
            "prevalence": lambda *_: 0.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0,
            "dwell_time": lambda *_: pd.Timedelta(days=DURATIONS.POSTPARTUM_DAYS),
        },
    )

    pregnant.add_dwell_time_transition(parturition)

    parturition.add_dwell_time_transition(postpartum)

    postpartum.add_dwell_time_transition(not_pregnant)

    return PregnancyModel(
        models.PREGNANCY_MODEL_NAME,
        states=[not_pregnant, pregnant, parturition, postpartum],
        get_data_functions={"cause_specific_mortality_rate": lambda *_: 0.0},
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

    @property
    def columns_required(self) -> List[str]:
        return ["pregnancy", "exit_time", "tracked"]

    def on_time_step_cleanup(self, event: Event) -> None:
        population = self.population_view.get(event.index)
        pop = population[
            (population["pregnancy"] == models.NOT_PREGNANT_STATE_NAME)
            & population["tracked"]
        ].copy()
        if len(pop) > 0:
            pop["tracked"] = pd.Series(False, index=pop.index)
            pop["exit_time"] = event.time
            self.population_view.update(pop)

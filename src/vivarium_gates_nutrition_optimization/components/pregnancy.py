import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.values import Pipeline
from vivarium_public_health.disease import (
    BaseDiseaseState,
    DiseaseModel,
    SusceptibleState,
)

from vivarium_gates_nutrition_optimization.components.children import NewChildren
from vivarium_gates_nutrition_optimization.components.disease import DiseaseState
from vivarium_gates_nutrition_optimization.constants import data_keys, models
from vivarium_gates_nutrition_optimization.constants.data_values import DURATIONS
from vivarium_gates_nutrition_optimization.constants.metadata import (
    ARTIFACT_INDEX_COLUMNS,
)


class NotPregnantState(SusceptibleState):
    def __init__(self, cause, *args, **kwargs):
        super(SusceptibleState, self).__init__(cause, *args, name_prefix="not_", **kwargs)


class PregnantState(DiseaseState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_children = NewChildren()
        self._sub_components += [self.new_children]

    @property
    def columns_created(self):
        return [
            self.event_time_column,
            self.event_count_column,
            "pregnancy_outcome",
            "pregnancy_duration",
        ] + self.new_children.columns_created

    def setup(self, builder: Builder):
        super(BaseDiseaseState, self).setup(builder)

        self.clock = builder.time.clock()

        view_columns = self.columns_created + [self._model, "alive"]
        self.population_view = builder.population.get_view(view_columns)
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.columns_created,
            requires_columns=[self._model] + ["intervention"],
        )
        self.prevalence = self.get_prevalence_table(builder)
        self.birth_prevalence = self.get_birth_prevalence_table(builder)
        self.dwell_time = self.get_dwell_time_pipeline(builder)
        self.has_disability, self.base_disability_weight = self.get_disability_weight_table(
            builder
        )
        self.disability_weight = self.get_disability_weight_pipeline(builder)
        self.register_disability_weight_modifier(builder)
        (
            self.has_excess_mortality,
            self.base_excess_mortality_rate,
        ) = self.get_excess_mortality_rate_table(builder)
        self.excess_mortality_rate = self.get_excess_mortality_rate_pipeline(builder)
        self.joint_paf = self.get_joint_paf_pipeline(builder)
        self.register_mortality_rate_modifier(builder)
        self.randomness_prevalence = self.get_prevalence_random_stream(builder)

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
        outcome_probabilities = self.birth_outcome_probabilities(pop_data.index)
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
                term_pop, self.new_children.columns_created + ["pregnancy_duration"]
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
    def setup(self, builder: Builder):
        """Perform this component's setup."""
        super(DiseaseModel, self).setup(builder)

        self.configuration_age_start = builder.configuration.population.age_start
        self.configuration_age_end = builder.configuration.population.age_end

        cause_specific_mortality_rate = self.load_cause_specific_mortality_rate_data(builder)
        self.cause_specific_mortality_rate = builder.lookup.build_table(
            cause_specific_mortality_rate,
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )
        builder.value.register_value_modifier(
            "cause_specific_mortality_rate",
            self.adjust_cause_specific_mortality_rate,
            requires_columns=["age", "sex"],
        )

        self.population_view = builder.population.get_view(["age", "sex", self.state_column])
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.state_column],
            requires_columns=["age", "sex"],
            requires_streams=[f"{self.state_column}_initial_states"],
        )
        self.randomness = builder.randomness.get_stream(f"{self.state_column}_initial_states")

        builder.event.register_listener("time_step", self.on_time_step, priority=3)
        builder.event.register_listener("time_step__cleanup", self.on_time_step_cleanup)


def Pregnancy():
    not_pregnant = NotPregnantState(models.PREGNANT_STATE_NAME)
    pregnant = PregnantState(
        models.PREGNANT_STATE_NAME,
        get_data_functions={
            "prevalence": lambda *_: 1.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0,
        },
    )
    parturition = DiseaseState(
        models.PARTURITION_STATE_NAME,
        get_data_functions={
            "prevalence": lambda *_: 0.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0,
            "dwell_time": lambda builder, cause: builder.time.step_size()(),
        },
    )
    postpartum = DiseaseState(
        models.POSTPARTUM_STATE_NAME,
        get_data_functions={
            "prevalence": lambda *_: 0.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0,
            "dwell_time": lambda *_: pd.Timedelta(days=DURATIONS.POSTPARTUM_DAYS),
        },
    )
    pregnant.allow_self_transitions()
    pregnant.add_transition(parturition)

    parturition.allow_self_transitions()
    parturition.add_transition(postpartum)

    postpartum.allow_self_transitions()
    postpartum.add_transition(not_pregnant)

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


class UntrackNotPregnant:
    """Component for untracking not pregnant simulants"""

    @property
    def name(self) -> str:
        return "untrack_not_pregnant"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.population_view = builder.population.get_view(
            ["pregnancy", "exit_time", "tracked"]
        )
        builder.event.register_listener("time_step__cleanup", self.on_time_step_cleanup)

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

    def __repr__(self) -> str:
        return "UntrackNotPregnant()"

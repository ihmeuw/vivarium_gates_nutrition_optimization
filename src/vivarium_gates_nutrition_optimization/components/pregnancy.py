import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.values import Pipeline
from vivarium_public_health.disease import DiseaseModel, SusceptibleState

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
    @property
    def columns_created(self):
        return [
            self.event_time_column,
            self.event_count_column,
            "pregnancy_term_outcome",
            "pregnancy_duration",
            # "pregnancy_birth_outcome",
        ]

    def setup(self, builder: Builder):
        super().setup(builder)
        self.time_step = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)
        self.partial_term_probs = builder.value.register_value_producer(
            "partial_term_probabilities",
            source=builder.lookup.build_table(
                get_partial_term_probability_data(builder),
                key_columns=["sex"],
                parameter_columns=["age", "year"],
            ),
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        for transition in self.transition_set:
            if transition.start_active:
                transition.set_active(pop_data.index)

        pop_events = self.get_initial_event_times(pop_data)
        pregnancy_term_outcomes_and_durations = self.sample_pregnancy_terms_and_durations(
            pop_data
        )
        pop_update = pd.concat([pop_events, pregnancy_term_outcomes_and_durations], axis=1)
        self.population_view.update(pop_update)

    def sample_pregnancy_terms_and_durations(self, pop_data: SimulantData) -> pd.DataFrame:
        term_outcome_probabilities = self.partial_term_probs(pop_data.index)
        term_outcome_probabilities = pd.DataFrame(
            {
                models.PARTIAL_TERM_OUTCOME: term_outcome_probabilities,
                models.FULL_TERM_OUTCOME: 1 - term_outcome_probabilities,
            }
        )
        pregnancy_term_outcomes = pd.DataFrame(
            {
                "pregnancy_term_outcome": self.randomness.choice(
                    pop_data.index,
                    choices=term_outcome_probabilities.columns.to_list(),
                    p=term_outcome_probabilities,
                    additional_key="pregnancy_term_outcome",
                )
            }
        )

        term_duration_map = {
            models.FULL_TERM_OUTCOME: lambda *_: pd.to_timedelta(
                DURATIONS.FULL_TERM, unit="days"
            ),
            models.PARTIAL_TERM_OUTCOME: self.sample_partial_term_durations,
        }

        for term_length, sampling_function in term_duration_map.items():
            term_pop = pregnancy_term_outcomes[
                pregnancy_term_outcomes["pregnancy_term_outcome"] == term_length
            ].index
            pregnancy_term_outcomes.loc[term_pop, "pregnancy_duration"] = sampling_function(
                term_pop
            )

        return pregnancy_term_outcomes

    def sample_partial_term_durations(self, partial_term_pop: pd.DataFrame) -> pd.Series:
        low, high = DURATIONS.DETECTION, DURATIONS.PARTIAL_TERM
        draw = self.randomness.get_draw(
            partial_term_pop, additional_key="partial_term_pregnancy_duration"
        )
        return pd.to_timedelta((low + (high - low) * draw), unit="days")

    def get_dwell_time_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            f"{self.state_id}.dwell_time",
            source=lambda index: self.population_view.get(index)["pregnancy_duration"],
            requires_columns=["age", "sex", "pregnancy_term_outcome"],
        )

    def get_initial_event_times(self, pop_data: SimulantData) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.event_time_column: self.clock() + self.time_step(),
                self.event_count_column: 1,
            },
            index=pop_data.index,
        )


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
    postpartum = DiseaseState(
        models.POSTPARTUM_STATE_NAME,
        get_data_functions={
            "prevalence": lambda *_: 0.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0,
            "dwell_time": lambda *_: pd.Timedelta(days=DURATIONS.POSTPARTUM),
        },
    )
    pregnant.allow_self_transitions()
    pregnant.add_transition(postpartum)

    postpartum.allow_self_transitions()
    postpartum.add_transition(not_pregnant)

    return DiseaseModel(
        models.PREGNANCY_MODEL_NAME,
        states=[not_pregnant, pregnant, postpartum],
        get_data_functions={"cause_specific_mortality_rate": lambda *_: 0.0},
    )


def get_partial_term_probability_data(builder: Builder) -> pd.DataFrame:
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

    return (
        (raw_incidence_miscarriage + raw_incidence_ectopic)
        / (
            asfr
            + asfr.multiply(sbr["value"], axis=0)
            + raw_incidence_miscarriage
            + raw_incidence_ectopic
        )
    ).reset_index()

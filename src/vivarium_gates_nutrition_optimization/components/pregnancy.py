import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

from vivarium_gates_nutrition_optimization.constants import data_keys, models
from vivarium_gates_nutrition_optimization.constants.data_values import (
    POSTPARTUM_DURATION,
)
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
            "pregnancy_term_length",
            "pregnancy_duration",
            "pregnancy_outcome",
        ]

    def setup(self, builder: Builder):
        super().setup(builder)
        self.time_step = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)
        self.partial_term_probs = builder.value.register_value_producer(
            "partial_term_probabilities",
            source=builder.lookup.build_table(
                get_partial_term_probabilities(builder),
                key_columns=['sex'],
                parameter_columns=['age', 'year'],
            )
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_events = self.get_initial_event_times(pop_data)
        pregnancy_term_outcome = self.sample_pregnancy_terms(pop_data)
        pop_update = pd.concat([pop_events, pregnancy_term_outcome], axis=1)
        self.population_view.update(pop_update)

    def sample_pregnancy_terms(self, pop_data: SimulantData) -> pd.DataFrame:
        p_term_outcome = self.partial_term_probs(pop_data.index)
        p_term_outcome = pd.DataFrame({models.PARTIAL_TERM_OUTCOME:p_term_outcome, models.FULL_TERM_OUTCOME: 1 - p_term_outcome})
        term_outcome = self.randomness.choice(
            pop_data.index,
            choices=p_term_outcome.columns.to_list(),
            p=p_term_outcome,
            additional_key='term_outcome'
        )
        return term_outcome

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
            "dwell_time": lambda *_: pd.Timedelta(days=40 * 7),
        },
    )
    postpartum = DiseaseState(
        models.POSTPARTUM_STATE_NAME,
        get_data_functions={
            "prevalence": lambda *_: 0.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0,
            "dwell_time": lambda *_: pd.Timedelta(days=POSTPARTUM_DURATION),
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


def get_partial_term_probabilities(builder: Builder) -> pd.DataFrame:
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

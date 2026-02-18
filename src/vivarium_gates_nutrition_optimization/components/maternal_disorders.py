from vivarium.framework.engine import Builder
from vivarium_public_health.disease import DiseaseModel, DiseaseState, RecoveredState
from vivarium_public_health.utilities import to_years

from vivarium_gates_nutrition_optimization.components.disease import (
    ParturitionSelectionState,
)
from vivarium_gates_nutrition_optimization.constants import data_keys, models
from vivarium_gates_nutrition_optimization.constants.metadata import (
    ARTIFACT_INDEX_COLUMNS,
)


def MaternalDisorders():
    cause = models.MATERNAL_DISORDERS_STATE_NAME
    susceptible = ParturitionSelectionState(cause, allow_self_transition=True)
    with_condition = DiseaseState(
        cause,
        allow_self_transition=True,
        prevalence=0.0,
        dwell_time=lambda builder: builder.time.step_size()(),
        disability_weight=lambda builder: get_maternal_disorders_disability_weight(builder),
        excess_mortality_rate=0.0,
    )
    recovered = RecoveredState(cause, allow_self_transition=True)
    susceptible.add_transition(with_condition)
    with_condition.add_dwell_time_transition(recovered)

    return DiseaseModel(
        cause,
        states=[susceptible, with_condition, recovered],
        cause_specific_mortality_rate=0.0,
    )


def MaternalHemorrhage():
    cause = models.MATERNAL_HEMORRHAGE_STATE_NAME
    susceptible = ParturitionSelectionState(cause, allow_self_transition=True)
    with_condition = DiseaseState(
        cause,
        allow_self_transition=True,
        prevalence=0.0,
        dwell_time=lambda builder, cause: builder.time.step_size()(),
        disability_weight=0.0,
        excess_mortality_rate=0.0,
    )
    recovered = RecoveredState(cause, allow_self_transition=True)
    susceptible.add_transition(with_condition)
    with_condition.add_dwell_time_transition(recovered)

    return DiseaseModel(
        cause,
        states=[susceptible, with_condition, recovered],
        cause_specific_mortality_rate=0.0,
    )


def get_maternal_disorders_disability_weight(builder: Builder):
    ylds = builder.data.load(data_keys.MATERNAL_DISORDERS.YLDS).set_index(
        ARTIFACT_INDEX_COLUMNS
    )
    timestep = builder.time.step_size()
    return ylds.div(to_years(timestep())).reset_index()

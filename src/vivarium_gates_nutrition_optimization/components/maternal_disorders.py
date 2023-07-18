from vivarium_public_health.disease import DiseaseModel, SusceptibleState, RecoveredState
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium_gates_nutrition_optimization.components.disease import DiseaseState, ParturitionSelectionState
from vivarium_gates_nutrition_optimization.constants import data_keys, models
from vivarium_gates_nutrition_optimization.constants.data_values import DURATIONS
from vivarium_gates_nutrition_optimization.constants.metadata import (
    ARTIFACT_INDEX_COLUMNS,
)
from vivarium_public_health.utilities import to_years



def MaternalDisorders():
    cause = models.MATERNAL_DISORDERS_STATE_NAME
    susceptible = ParturitionSelectionState(cause)
    with_condition = DiseaseState(
        cause,
        get_data_functions={
            "prevalence": lambda *_: 0.0,
            "disability_weight": get_maternal_disorders_disability_weight,
            "dwell_time":lambda builder, cause: builder.time.step_size()(),
        },
    )
    recovered = RecoveredState(cause)

    susceptible.allow_self_transitions()
    susceptible.add_transition(
        with_condition, source_data_type="rate") 
    with_condition.allow_self_transitions()
    with_condition.add_transition(
        recovered)
    recovered.allow_self_transitions()

    return DiseaseModel(
        cause,
        states=[susceptible, with_condition, recovered],
    )


def get_maternal_disorders_disability_weight(builder: Builder, cause: str):
    ylds = builder.data.load(data_keys.MATERNAL_DISORDERS.YLDS).set_index(ARTIFACT_INDEX_COLUMNS)
    timestep = builder.time.step_size()
    return ylds.div(to_years(timestep())).reset_index()
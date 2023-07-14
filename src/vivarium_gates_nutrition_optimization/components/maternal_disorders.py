import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.values import Pipeline
from vivarium_public_health.disease import DiseaseModel, SusceptibleState, RecoveredState

from vivarium_gates_nutrition_optimization.components.disease import DiseaseState
from vivarium_gates_nutrition_optimization.constants import data_keys, models
from vivarium_gates_nutrition_optimization.constants.data_values import DURATIONS
from vivarium_gates_nutrition_optimization.constants.metadata import (
    ARTIFACT_INDEX_COLUMNS,
)

def MaternalDisorders():
    cause = models.MATERNAL_DISORDERS_STATE_NAME
    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause)
    recovered = RecoveredState(cause)

    healthy.allow_self_transitions()
    healthy.add_transition(infected, source_data_type="rate")
    infected.allow_self_transitions()
    infected.add_transition(recovered, source_data_type="rate")
    recovered.allow_self_transitions()

    return DiseaseModel(cause, states=[healthy, infected, recovered])
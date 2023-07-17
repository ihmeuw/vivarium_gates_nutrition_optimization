from vivarium_public_health.disease import DiseaseModel, SusceptibleState, RecoveredState
import numpy as np
from vivarium_gates_nutrition_optimization.components.disease import DiseaseState, ParturitionSelectionState
from vivarium_gates_nutrition_optimization.constants import data_keys, models
from vivarium_gates_nutrition_optimization.constants.data_values import DURATIONS


def MaternalDisorders():
    cause = models.MATERNAL_DISORDERS_STATE_NAME
    susceptible = ParturitionSelectionState(cause)
    with_condition = DiseaseState(
        cause,
        get_data_functions={
            "prevalence": lambda *_: 0.0,
            "disability_weight": lambda *_: 0.0,  ## Add proper DW here--get through artifact?
            "excess_mortality_rate": lambda *_: 0.0,  ## add proper EMR here -- get through artifact?
            "dwell_time": lambda *_: DURATIONS.PARTURITION, ## Do this as a timestep instead
        },
    )
    recovered = RecoveredState(cause)

    susceptible.allow_self_transitions()
    susceptible.add_transition(
        with_condition, source_data_type="rate", get_data_functions={"incidence_rate": lambda *_:365 / 7 * -np.log(1 - 0.75)}
    ) 
    with_condition.allow_self_transitions()
    with_condition.add_transition(
        recovered)
    recovered.allow_self_transitions()

    return DiseaseModel(
        cause,
        states=[susceptible, with_condition, recovered],
        get_data_functions={
            "cause_specific_mortality_rate": lambda *_: 0.0,
        },
    )
    ## Add CSMR here--get through artifact?

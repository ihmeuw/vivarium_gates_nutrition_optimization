from vivarium_public_health.disease import DiseaseModel, SusceptibleState, RecoveredState

from vivarium_gates_nutrition_optimization.components.disease import DiseaseState
from vivarium_gates_nutrition_optimization.constants import data_keys, models
from vivarium_gates_nutrition_optimization.constants.data_values import DURATIONS


def MaternalDisorders():
    cause = models.MATERNAL_DISORDERS_STATE_NAME
    healthy = SusceptibleState(cause)
    infected = DiseaseState(
        cause,
        get_data_functions={
            "prevalence": lambda *_: 0.0,
            "disability_weight": lambda *_: 0.0,  ## Add proper DW here--get through artifact?
            "excess_mortality_rate": lambda *_: 0.0,  ## add proper EMR here -- get through artifact?
            "dwell_time": lambda *_: DURATIONS.MATERNAL_DISORDERS,
        },
    )
    recovered = RecoveredState(cause)

    healthy.allow_self_transitions()
    healthy.add_transition(
        infected, source_data_type="rate", get_data_functions={"incidence_rate": lambda *_: 0.75}
    ) 
    infected.allow_self_transitions()
    infected.add_transition(
        recovered)
    recovered.allow_self_transitions()

    return DiseaseModel(
        cause,
        states=[healthy, infected, recovered],
        get_data_functions={
            "cause_specific_mortality_rate": lambda *_: 0.0,
        },
    )
    ## Add CSMR here--get through artifact?

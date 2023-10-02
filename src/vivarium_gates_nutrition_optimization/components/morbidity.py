from vivarium_public_health.disease import DiseaseModel, SusceptibleState

from vivarium_gates_nutrition_optimization.components.disease import (
    ParturitionExclusionState,
)


def BackgroundMorbidity():
    cause = "other_causes"
    susceptible = SusceptibleState(cause)
    with_condition = ParturitionExclusionState(
        cause,
        allow_self_transition=True,
        get_data_functions={
            "prevalence": lambda *_: 1.0,
            "excess_mortality_rate": lambda *_: 0.0,
        },
    )
    return DiseaseModel(
        cause,
        states=[susceptible, with_condition],
        get_data_functions={"cause_specific_mortality_rate": lambda *_: 0.0},
    )

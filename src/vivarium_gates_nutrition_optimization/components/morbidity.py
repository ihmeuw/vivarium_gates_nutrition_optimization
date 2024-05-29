from vivarium_public_health.disease import DiseaseModel, SusceptibleState

from vivarium_gates_nutrition_optimization.components.disease import (
    ParturitionExclusionState,
)


def BackgroundMorbidity():
    # NOTE: I have not updated this component to work with the new lookup table
    # configuration work so this component is not used in the model spec because
    # it is currently broken. This component was an exploratory component and may
    # be revisited in the future - albrja
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

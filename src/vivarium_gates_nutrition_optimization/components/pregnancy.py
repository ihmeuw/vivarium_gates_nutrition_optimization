from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

from vivarium_gates_nutrition_optimization.constants import data_keys, models


def Pregnancy():
    not_pregnant = SusceptibleState(models.PREGNANCY_STATE)
    pregnant = DiseaseState(
        models.PREGNANCY_STATE,
        get_data_functions={
            "prevalence": lambda *_: 1.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0,
        },
    )
    pregnant.allow_self_transitions()
    return DiseaseModel(
        models.PREGNANCY_MODEL_NAME,
        states=[not_pregnant, pregnant],
        get_data_functions={"cause_specific_mortality_rate": lambda *_: 0.0},
    )

import pandas as pd
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState, RecoveredState

from vivarium_gates_nutrition_optimization.constants import data_keys, models


class NotPregnantState(SusceptibleState):
    def __init__(self, cause, *args, **kwargs):
        super(SusceptibleState, self).__init__(cause, *args, name_prefix="not_", **kwargs)

def Pregnancy():
    not_pregnant = NotPregnantState(models.PREGNANCY_STATE)
    pregnant = DiseaseState(
        models.PREGNANCY_STATE,
        get_data_functions={
            "prevalence": lambda *_: 1.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0, 
            "dwell_time": lambda *_: pd.Timedelta(days=40*7)
        },
    )
    postpartum = DiseaseState(models.POSTPARTUM_STATE_NAME,
                                      get_data_functions={
            "prevalence": lambda *_: 1.0,
            "disability_weight": lambda *_: 0.0,
            "excess_mortality_rate": lambda *_: 0.0, 
        },
                              )
    pregnant.allow_self_transitions()
    pregnant.add_transition(postpartum)
    postpartum.allow_self_transitions()
    return DiseaseModel(
        models.PREGNANCY_MODEL_NAME,
        states=[not_pregnant, pregnant, postpartum],
        get_data_functions={"cause_specific_mortality_rate": lambda *_: 0.0},
    )

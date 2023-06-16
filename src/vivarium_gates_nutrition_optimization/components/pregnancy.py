import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

from vivarium_gates_nutrition_optimization.constants import models


class NotPregnantState(SusceptibleState):
    def __init__(self, cause, *args, **kwargs):
        super(SusceptibleState, self).__init__(cause, *args, name_prefix="not_", **kwargs)


class PregnantState(DiseaseState):
    def setup(self, builder: Builder):
        super().setup(builder)
        self.time_step = builder.time.step_size()

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
            "dwell_time": lambda *_: pd.Timedelta(days=6 * 7),
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

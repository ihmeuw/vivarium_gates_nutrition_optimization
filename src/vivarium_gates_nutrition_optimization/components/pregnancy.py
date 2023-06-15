import pandas as pd
from vivarium.framework.population import SimulantData
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

from vivarium_gates_nutrition_optimization.constants import models


class NotPregnantState(SusceptibleState):
    def __init__(self, cause, *args, **kwargs):
        super(SusceptibleState, self).__init__(cause, *args, name_prefix="not_", **kwargs)


class PregnantState(DiseaseState):
    def get_initial_event_times(self, pop_data: SimulantData) -> pd.DataFrame:
        pop_update = super().get_initial_event_times(pop_data)

        simulants_with_condition = self.population_view.subview([self._model]).get(
            pop_data.index, query=f'{self._model}=="{self.state_id}"'
        )
        if not simulants_with_condition.empty:
            pop_update.loc[
                simulants_with_condition.index, self.event_time_column
            ] = self.clock()

        return pop_update


def Pregnancy():
    not_pregnant = NotPregnantState(models.PREGNANCY_STATE)
    pregnant = PregnantState(
        models.PREGNANCY_STATE,
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

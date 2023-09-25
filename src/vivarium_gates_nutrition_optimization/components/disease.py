from typing import Callable, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import State, Transition
from vivarium.framework.values import Pipeline
from vivarium_public_health.disease import DiseaseState 
from vivarium_public_health.disease import SusceptibleState
from vivarium_public_health.disease.transition import ProportionTransition

from vivarium_gates_nutrition_optimization.constants import models

class ParturitionSelectionState(SusceptibleState):
    def add_transition(
        self,
        output: State,
        source_data_type: str = "proportion",
        get_data_functions: Dict[str, Callable] = None,
        **kwargs,
    ) -> Transition:
        transition = ParturitionSelectionTransition(
            self,
            output,
            get_data_functions={
                "proportion": lambda builder, cause: builder.data.load(
                    f"cause.{cause}.incident_probability"
                )
            },
            **kwargs,
        )
        self.transition_set.append(transition)
        return transition


class ParturitionSelectionTransition(ProportionTransition):
    def setup(self, builder: Builder):
        super().setup(builder)

        pipeline_name = f"{self.output_state.cause}.transition_proportion"
        self.proportion_pipeline = builder.value.register_value_producer(
            pipeline_name,
            source=self.compute_transition_proportion,
            requires_columns=["age", "sex", "alive"],
        )
        self.population_view = builder.population.get_view(["alive", "pregnancy"])

    def compute_transition_proportion(self, index):
        transition_proportion = pd.Series(0.0, index=index)
        sub_pop = self.population_view.get(
            index, query="(alive == 'alive') & (pregnancy == 'parturition')"
        ).index

        transition_proportion.loc[sub_pop] = self.proportion(sub_pop)
        return transition_proportion

    def _probability(self, index):
        return self.proportion_pipeline(index)


class ParturitionExclusionState(DiseaseState):
    def setup(self, builder: Builder):
        super().setup(builder)
        view_columns = self.columns_created + [self._model, "alive", "pregnancy", "tracked"]
        self.population_view = builder.population.get_view(view_columns)

    def get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            f"{self.state_id}.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=["age", "sex", "alive", "pregnancy", self._model],
        )

    def compute_disability_weight(self, index: pd.Index):
        disability_weight = raw_disability_weight = pd.Series(0, index=index)
        with_condition = self.with_condition(index)
        raw_disability_weight.loc[with_condition] = self.base_disability_weight(
            with_condition
        )

        dw_map = {
            models.NOT_PREGNANT_STATE_NAME: raw_disability_weight,
            models.PREGNANT_STATE_NAME: raw_disability_weight,
            ## Pause YLD accumulation during the parturition state
            models.PARTURITION_STATE_NAME: pd.Series(0, index=index),
            models.POSTPARTUM_STATE_NAME: raw_disability_weight,
        }

        pop = self.population_view.get(index)
        alive = pop["alive"] == "alive"
        for state, dw in dw_map.items():
            in_state = alive & (pop["pregnancy"] == state)
            disability_weight[in_state] = dw.loc[in_state]

        return disability_weight

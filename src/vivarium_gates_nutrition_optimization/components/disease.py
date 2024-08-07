from typing import Callable, Dict, List

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import State, Transition
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor
from vivarium_public_health.disease import DiseaseState, SusceptibleState
from vivarium_public_health.disease.transition import ProportionTransition
from vivarium_public_health.utilities import get_lookup_columns

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
    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> List[str]:
        return ["alive", "pregnancy"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        pipeline_name = f"{self.output_state.state_id}.transition_proportion"
        self.proportion_pipeline = builder.value.register_value_producer(
            pipeline_name,
            source=self.compute_transition_proportion,
            requires_columns=["age", "sex", "alive"],
        )

    ###################
    # Pipeline methods#
    ###################

    def compute_transition_proportion(self, index) -> pd.Series:
        transition_proportion = pd.Series(0.0, index=index)
        sub_pop = self.population_view.get(
            index, query="(alive == 'alive') & (pregnancy == 'parturition')"
        ).index

        transition_proportion.loc[sub_pop] = self.lookup_tables["proportion"](sub_pop)
        return transition_proportion

    ####################
    # Helper methods   #
    ####################

    def _probability(self, index) -> pd.Series:
        return self.proportion_pipeline(index)


class ParturitionExclusionState(DiseaseState):
    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> List[str]:
        return super().columns_required + ["pregnancy", "tracked"]

    # #####################
    # # Lifecycle methods #
    # #####################

    def get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
        lookup_columns = get_lookup_columns([self.lookup_tables["disability_weight"]])
        return builder.value.register_value_producer(
            f"{self.state_id}.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=lookup_columns + ["alive", self.model, "pregnancy"],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def compute_disability_weight(self, index: pd.Index) -> pd.Series:
        disability_weight = pd.Series(0, index=index)
        raw_disability_weight = pd.Series(0, index=index)
        with_condition = self.with_condition(index)
        # FIXME: this is broken (self.base_disability_weight is undefined)
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

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import State, Transition
from vivarium_public_health.disease import DiseaseState, SusceptibleState
from vivarium_public_health.disease.transition import ProportionTransition

from vivarium_gates_nutrition_optimization.constants import models


class ParturitionSelectionState(SusceptibleState):
    def add_transition(
        self,
        output: State,
        source_data_type: str = "proportion",
        **kwargs,
    ) -> Transition:
        transition = ParturitionSelectionTransition(
            self,
            output,
            proportion=f"cause.{output.state_id}.incident_probability",
            **kwargs,
        )
        self.transition_set.append(transition)
        return transition


class ParturitionSelectionTransition(ProportionTransition):

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.pipeline_name = f"{self.output_state.state_id}.transition_proportion"
        builder.value.register_attribute_producer(
            self.pipeline_name,
            source=self.compute_transition_proportion,
            required_resources=["age", "sex", "is_alive"],
        )

    ###################
    # Pipeline methods#
    ###################

    def compute_transition_proportion(self, index) -> pd.Series:
        transition_proportion = pd.Series(0.0, index=index)
        sub_pop_idx = self.population_view.get_filtered_index(
            index, query="(is_alive == True) & (pregnancy == 'parturition')"
        )

        transition_proportion.loc[sub_pop_idx] = self.proportion_table(sub_pop_idx)
        return transition_proportion

    ####################
    # Helper methods   #
    ####################

    def _probability(self, index) -> pd.Series:
        return self.population_view.get(index, self.pipeline_name)


# NOTE: This component is only used by Morbidity which is an exploratory component
#   and not fully functional. At the very least, this class has not been updated
#   to work with the vivarium v4.0 / vph v5.0. Commenting it out.
# class ParturitionExclusionState(DiseaseState):

#     # #####################
#     # # Lifecycle methods #
#     # #####################

#     def get_disability_weight_pipeline(self, builder: Builder) -> None:
#         builder.value.register_attribute_producer(
#             f"{self.state_id}.disability_weight",
#             source=self.compute_disability_weight,
#             required_resources=[
#                 "is_alive",
#                 "pregnancy",
#                 self.model,
#                 self.lookup_tables["disability_weight"],
#             ],
#         )

#     ##################################
#     # Pipeline sources and modifiers #
#     ##################################

#     def compute_disability_weight(self, index: pd.Index) -> pd.Series:
#         disability_weight = pd.Series(0, index=index)
#         raw_disability_weight = pd.Series(0, index=index)
#         with_condition = self.with_condition(index)
#         # FIXME: this is broken (self.base_disability_weight is undefined)
#         raw_disability_weight.loc[with_condition] = self.base_disability_weight(
#             with_condition
#         )

#         dw_map = {
#             models.NOT_PREGNANT_STATE_NAME: raw_disability_weight,
#             models.PREGNANT_STATE_NAME: raw_disability_weight,
#             ## Pause YLD accumulation during the parturition state
#             models.PARTURITION_STATE_NAME: pd.Series(0, index=index),
#             models.POSTPARTUM_STATE_NAME: raw_disability_weight,
#         }

#         pop = self.population_view.get(index)
#         alive = pop["is_alive"] == True
#         for state, dw in dw_map.items():
#             in_state = alive & (pop["pregnancy"] == state)
#             disability_weight[in_state] = dw.loc[in_state]

#         return disability_weight

from typing import Callable, Dict, List

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import State, Transition
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor
from vivarium_public_health.disease import DiseaseState, SusceptibleState
from vivarium_public_health.disease.transition import ProportionTransition
from vivarium_public_health.utilities import is_non_zero

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

        transition_proportion.loc[sub_pop] = self.proportion(sub_pop)
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
        super().columns_required + ["pregnancy", "tracked"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super(DiseaseState, self).setup(builder)
        self.clock = builder.time.clock()

        prevalence_data = self.load_prevalence_data(builder)
        self.prevalence = builder.lookup.build_table(
            prevalence_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

        birth_prevalence_data = self.load_birth_prevalence_data(builder)
        self.birth_prevalence = builder.lookup.build_table(
            birth_prevalence_data, key_columns=["sex"], parameter_columns=["year"]
        )

        dwell_time_data = self.load_dwell_time_data(builder)
        self.dwell_time = builder.value.register_value_producer(
            f"{self.state_id}.dwell_time",
            source=builder.lookup.build_table(
                dwell_time_data, key_columns=["sex"], parameter_columns=["age", "year"]
            ),
            requires_columns=["age", "sex"],
        )

        disability_weight_data = self.load_disability_weight_data(builder)
        self.has_disability = is_non_zero(disability_weight_data)
        self.base_disability_weight = builder.lookup.build_table(
            disability_weight_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.disability_weight = builder.value.register_value_producer(
            f"{self.state_id}.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=["age", "sex", "alive", self.model, "pregnancy"],
        )
        builder.value.register_value_modifier(
            "disability_weight", modifier=self.disability_weight
        )

        excess_mortality_data = self.load_excess_mortality_rate_data(builder)
        self.has_excess_mortality = is_non_zero(excess_mortality_data)
        self.base_excess_mortality_rate = builder.lookup.build_table(
            excess_mortality_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.excess_mortality_rate = builder.value.register_rate_producer(
            self.excess_mortality_rate_pipeline_name,
            source=self.compute_excess_mortality_rate,
            requires_columns=["age", "sex", "alive", self.model],
            requires_values=[self.excess_mortality_rate_paf_pipeline_name],
        )
        paf = builder.lookup.build_table(0)
        self.joint_paf = builder.value.register_value_producer(
            self.excess_mortality_rate_paf_pipeline_name,
            source=lambda idx: [paf(idx)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )
        builder.value.register_value_modifier(
            "mortality_rate",
            modifier=self.adjust_mortality_rate,
            requires_values=[self.excess_mortality_rate_pipeline_name],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def compute_disability_weight(self, index: pd.Index) -> pd.Series:
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

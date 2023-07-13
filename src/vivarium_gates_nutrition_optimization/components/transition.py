from typing import Callable, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.state_machine import State, Transition
from vivarium.framework.utilities import rate_to_probability
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor


class CompositeRateTransition(Transition):
    """
    A component that manages transitions from a single state to multiple output
    states.
    """

    def __init__(self, input_state, output_state, **kwargs):
        super().__init__(
            input_state, output_state, probability_func=self._probability, **kwargs
        )

        # A dictionary with output state name as the key and the
        # get_data_functions for the transition to that state as its value
        self._sub_transition_sources: Dict[str, Dict[str, Callable]] = {}

    def __str__(self):
        return f"CompositeRateTransition(from={self.input_state.state_id})"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self._pipeline_state_map = {}

        self._lookup_tables = self._create_lookup_tables(builder)
        self.transition_pipelines = self._register_transition_pipelines(builder)
        self.transition_pafs = self._register_paf_pipelines(builder)

        # registering value producer not a rate producer, because the sub rates
        # are already scaled to the time-step
        self.get_transition_rate = builder.value.register_value_producer(
            f"{self.input_state.state_id}.composite_exit_rate",
            source=self.compute_transition_rate,
            requires_values=[
                pipeline.name for pipeline in self.transition_pipelines.values()
            ],
        )

        self.population_view = builder.population.get_view(["alive"])

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def compute_transition_rate(self, index: pd.Index):
        """
        Calculates the exit rate from the input state. This is the sum of each
        of the subsidiary rates.
        """
        rates = pd.concat(
            [rate(index) for rate in self.transition_pipelines.values()], axis=1
        )
        return rates.sum(axis=1)

    ##################
    # Public methods #
    ##################

    def add_transition(
        self, output_state_name: str, get_data_functions: Dict[str, Callable] = None
    ) -> None:
        """
        Registers a rate to a new output state with the CompositeRateTransition
        """
        self._sub_transition_sources[output_state_name] = get_data_functions

    def get_rate_to_state(self, index: pd.Index, output_state: State) -> pd.Series:
        """Gets the rate from the input state to a specific output state"""
        return self.transition_pipelines[output_state.state_id](index)

    ##################
    # Helper methods #
    ##################

    def _probability(self, index):
        """
        Converts the rate from the input state to the transient state to a
        probability.
        """
        return pd.Series(rate_to_probability(self.get_transition_rate(index)))

    def _create_lookup_tables(
        self,
        builder: Builder,
    ) -> Dict[str, LookupTable]:
        """
        Creates LookupTables for each subsidiary transition rate registered and
        stores them in a dictionary with the desired pipeline name as the key
        and the LookupTable as the value.

        Also creates mapping from pipeline name to output state name.

        Pipeline names use the same convention as the transitions created by
        standard VPH transitions.
        """
        lookup_tables = {}
        for output_state_name, get_data_functions in self._sub_transition_sources.items():
            if "incidence_rate" in get_data_functions:
                rate_data = get_data_functions["incidence_rate"](builder, output_state_name)
                pipeline_name = f"{output_state_name}.incidence_rate"
            elif "remission_rate" in get_data_functions:
                rate_data = get_data_functions["remission_rate"](builder, output_state_name)
                pipeline_name = f"{output_state_name}.remission_rate"
            elif "transition_rate" in get_data_functions:
                rate_data = get_data_functions["transition_rate"](
                    builder, self.input_state.state_id, output_state_name
                )
                pipeline_name = (
                    f"{self.input_state.state_id}_to_{output_state_name}.transition_rate"
                )
            else:
                raise ValueError("No valid data functions supplied.")

            lookup_table = builder.lookup.build_table(
                rate_data, key_columns=["sex"], parameter_columns=["age", "year"]
            )
            lookup_tables[pipeline_name] = lookup_table
            self._pipeline_state_map[pipeline_name] = output_state_name
        return lookup_tables

    def _register_transition_pipelines(
        self,
        builder: Builder,
    ) -> Dict[str, Pipeline]:
        """
        Registers all transition pipelines and stores them in a dictionary with
        the output state as the key and the pipeline as the value.
        """
        return {
            self._pipeline_state_map[pipeline_name]: builder.value.register_rate_producer(
                pipeline_name,
                source=self._get_pipeline_source(pipeline_name),
                requires_columns=["age", "sex", "alive"],
                requires_values=[f"{pipeline_name}.paf"],
            )
            for pipeline_name in self._lookup_tables
        }

    def _get_pipeline_source(self, pipeline_name: str) -> Callable[[pd.Index], pd.Series]:
        """
        Gets the function to be used as the source of the pipeline with the
        provided name.
        """

        def compute_transition_rate(index: pd.Index) -> pd.Series:
            """Gets the transition rate for each simulant in the given index"""
            transition_rate = pd.Series(0, index=index)
            living = self.population_view.get(index, query='alive == "alive"').index
            base_rates = self._lookup_tables[pipeline_name](living)
            joint_paf = self.transition_pafs[pipeline_name](living)
            transition_rate.loc[living] = base_rates * (1 - joint_paf)
            return transition_rate

        return compute_transition_rate

    def _register_paf_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        """Registers paf pipelines for all transitions"""
        return {
            pipeline_name: builder.value.register_value_producer(
                f"{pipeline_name}.paf",
                source=lambda index: [pd.Series(0, index=index)],
                preferred_combiner=list_combiner,
                preferred_post_processor=union_post_processor,
            )
            for pipeline_name in self._lookup_tables
        }

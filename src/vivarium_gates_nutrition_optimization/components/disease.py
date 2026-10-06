import pandas as pd
from vivarium.engine.framework.engine import Builder
from vivarium.engine.framework.state_machine import State, Transition
from vivarium.public_health.disease import SusceptibleState
from vivarium.public_health.disease.transition import ProportionTransition


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
            description=(
                "The probability that a simulant in parturition develops this maternal "
                "outcome, its artifact incident probability; zero for everyone else"
            ),
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

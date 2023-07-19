import pandas as pd
from vivarium.framework.engine import Builder
from vivarium_public_health.disease.transition import ProportionTransition
from vivarium_gates_nutrition_optimization.constants import data_keys


class ParturitionSelectionTransition(ProportionTransition):
    def setup(self, builder: Builder):
        super().setup(builder)

        pipeline_name = (
            f"{self.input_state.cause}_to_{self.output_state.cause}.transition_proportion"
        )
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
    

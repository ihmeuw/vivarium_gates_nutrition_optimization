import pandas as pd
from vivarium.framework.engine import Builder
from vivarium_public_health.disease.transition import ProportionTransition


class ParturitionSelectionTransition(ProportionTransition):
    def setup(self, builder: Builder):
        super().setup(builder)

        pipeline_name = (
            f"{self.input_state.cause}_to_{self.output_state.cause}.transition_rate"
        )
        self.transition_proportion = builder.value.register_value_producer(
            pipeline_name,
            source=self.compute_transition_proportion,
            requires_columns=["age", "sex", "alive"],
            requires_values=[f"{pipeline_name}.paf"],
        )

    def compute_transition_proporiton(self, index):
        transition_proportion = pd.Series(0, index=index)
        sub_pop = self.population_view.get(
            index, query="(alive == 'alive') & (pregnancy == 'parturition')"
        ).index

        transition_proportion.loc[sub_pop] = self.transition_proportion(sub_pop)
        return transition_proportion

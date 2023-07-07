from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium_public_health.metrics import DiseaseObserver
from vivarium_public_health.metrics import ResultsStratifier as ResultsStratifier_
from vivarium_public_health.metrics.stratification import Source, SourceType
from vivarium_public_health.utilities import to_years

from vivarium_gates_nutrition_optimization.constants import models


class ResultsStratifier(ResultsStratifier_):
    def register_stratifications(self, builder: Builder) -> None:
        super().register_stratifications(builder)

        self.setup_stratification(
            builder,
            name="pregnancy_outcome",
            sources=[Source("pregnancy_outcome", SourceType.COLUMN)],
            categories=models.PREGNANCY_OUTCOMES,
        )


class PregnancyObserver(DiseaseObserver):
    def __init__(self):
        super().__init__("pregnancy")

    def _get_population_view(self, builder: Builder) -> PopulationView:
        columns_required = [
            self.current_state_column_name,
            self.previous_state_column_name,
            "pregnancy_outcome",
        ]
        return builder.population.get_view(columns_required)

    # def on_collect_metrics(self, event: Event) -> None:
    #     pop = self.population_view.get(
    #         event.index, query='tracked == True and alive == "alive"'
    #     )
    #     new_observations = {}
    #     groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
    #     for label, group_mask in groups:
    #         for transition in self.transitions:
    #             key = f"{self.disease}_{transition}_event_count_{label}"
    #             transition_mask = (
    #                 group_mask
    #                 & (pop[self.previous_state_column_name] == transition.from_state)
    #                 & (pop[self.current_state_column_name] == transition.to_state)
    #             )
    #             new_observations[key] = transition_mask.sum()

    #     self.counts.update(new_observations)

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium_public_health.metrics import DiseaseObserver, MortalityObserver
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
    
class MaternalMortalityObserver(MortalityObserver):

    def on_post_setup(self, event: Event) -> None:
        self.causes_of_death += ["maternal_disorders"]

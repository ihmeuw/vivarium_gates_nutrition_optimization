from vivarium_public_health.population import BasePopulation
from vivarium_public_health.population.base_population import Disability

from vivarium_gates_nutrition_optimization.components.mortality import MaternalMortality
from vivarium_gates_nutrition_optimization.components.pregnancy import (
    UntrackNotPregnant,
)


class Population(BasePopulation):
    """Use BasePopulation except do not include Mortality as a sub-component."""

    def __init__(self):
        super().__init__()
        self._sub_components = [
            Disability(),
            # MaternalMortality(),
            # UntrackNotPregnant(),
        ]

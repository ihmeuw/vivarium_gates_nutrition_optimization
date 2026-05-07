from typing import Any, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health.population import Mortality

from vivarium_gates_nutrition_optimization.constants import data_keys


class MaternalMortality(Mortality):
    ##############
    # Properties #
    ##############

    @property
    def time_step_priority(self) -> int:
        return 9

    @property
    def configuration_defaults(self) -> Dict[str, Dict[str, Any]]:
        return {
            self.name: {
                "data_sources": {
                    "life_expectancy": "population.theoretical_minimum_risk_life_expectancy",
                    "all_cause_mortality_rate": 0.0,
                    "unmodeled_cause_specific_mortality_rate": 0.0,
                },
            },
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self.mortality_probability_name = "mortality_probability"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.register_mortality_probability(builder)

    ###################
    # Setup Methods   #
    ###################

    def register_mortality_probability(self, builder: Builder):
        # NOTE: I did not add this to the configurable lookup tables because
        # it is only used as the source for the pipeline.
        probability_data = builder.data.load(
            data_keys.MATERNAL_DISORDERS.MORTALITY_PROBABILITY
        )
        probability_pipeline_source = self.build_lookup_table(
            builder,
            "mortality_probability_source",
            probability_data,
        )
        builder.value.register_attribute_producer(
            self.mortality_probability_name, source=probability_pipeline_source
        )

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        pop_idx = self.population_view.get_filtered_index(
            event.index,
            query="(is_alive == True) & (maternal_disorders == 'maternal_disorders')",
        )
        mortality_probability = self.population_view.get(
            pop_idx, self.mortality_probability_name
        )

        deaths = self.random.filter_for_probability(
            pop_idx, mortality_probability, additional_key="death"
        )

        self.population_view.update(
            [
                "is_alive",
                self.years_of_life_lost_column_name,
                self.cause_of_death_column_name,
            ],
            lambda _: pd.DataFrame(
                {
                    "is_alive": False,
                    self.years_of_life_lost_column_name: self.life_expectancy_table(deaths),
                    self.cause_of_death_column_name: "maternal_disorders",
                },
                index=deaths,
            ),
        )

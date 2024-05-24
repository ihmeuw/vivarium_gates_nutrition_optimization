from typing import Any, Dict, List, Union

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline
from vivarium_public_health.population import Mortality
from vivarium_public_health.utilities import get_lookup_columns

from vivarium_gates_nutrition_optimization.constants import data_keys


class MaternalMortality(Mortality):
    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> List[str]:
        return super().columns_required + ["maternal_disorders", "pregnancy"]

    @property
    def time_step_priority(self) -> int:
        return 9

    @property
    def configuration_defaults(self) -> Dict[str, Dict[str, Any]]:
        return {
            self.name: {
                "data_sources": {
                    "life_expectancy": "population.theoretical_minimum_risk_life_expectancy",
                },
            },
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self.mortality_probability_pipeline_name = "mortality_probability"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.random = self.get_randomness_stream(builder)
        self.clock = builder.time.clock()
        self.mortality_probability = self.get_mortality_probability(builder)

    ###################
    # Setup Methods   #
    ###################

    def get_mortality_probability(self, builder: Builder):
        probability_data = builder.data.load(
            data_keys.MATERNAL_DISORDERS.MORTALITY_PROBABILITY
        )
        probability_pipeline_source = self.build_lookup_table(
            builder,
            probability_data,
        )
        return builder.value.register_value_producer(
            self.mortality_probability_pipeline_name,
            source=probability_pipeline_source,
            requires_columns=get_lookup_columns([probability_pipeline_source]),
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                self.cause_of_death_column_name: "not_dead",
                self.years_of_life_lost_column_name: 0.0,
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(
            event.index,
            query="(alive == 'alive') & (maternal_disorders == 'maternal_disorders')",
        )
        mortality_probability = self.mortality_probability(pop.index)

        deaths = self.random.filter_for_probability(
            pop.index, mortality_probability, additional_key="death"
        )

        pop.loc[deaths, "alive"] = "dead"
        pop.loc[deaths, "exit_time"] = event.time
        pop.loc[deaths, "years_of_life_lost"] = self.lookup_tables["life_expectancy"](deaths)
        pop.loc[deaths, "cause_of_death"] = "maternal_disorders"
        self.population_view.update(pop)

from typing import Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline

from vivarium_gates_nutrition_optimization.constants import data_keys


class MaternalMortality:
    def __init__(self):
        self._randomness_stream_name = "mortality_handler"

        self.cause_of_death_column_name = "cause_of_death"
        self.years_of_life_lost_column_name = "years_of_life_lost"

        self.mortality_probability_pipeline_name = "mortality_probability"

    @property
    def name(self) -> str:
        return "maternal_mortality"

    def __repr__(self) -> str:
        return f"MaternalMortality()"

        # noinspection PyAttributeOutsideInit

    def setup(self, builder: Builder) -> None:
        self.random = self._get_randomness_stream(builder)
        self.clock = builder.time.clock()

        self.mortality_probability = self._get_mortality_probability(builder)

        self.life_expectancy = self._get_life_expectancy(builder)

        self.population_view = self._get_population_view(builder)

        self._register_simulant_initializer(builder)
        self._register_on_timestep_listener(builder)

    def _get_randomness_stream(self, builder: Builder) -> RandomnessStream:
        return builder.randomness.get_stream(self._randomness_stream_name)

        # noinspection PyMethodMayBeStatic

    def _get_life_expectancy(self, builder: Builder) -> Union[LookupTable, Pipeline]:
        life_expectancy_data = builder.data.load(
            "population.theoretical_minimum_risk_life_expectancy"
        )
        return builder.lookup.build_table(life_expectancy_data, parameter_columns=["age"])

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            [
                self.cause_of_death_column_name,
                self.years_of_life_lost_column_name,
                "alive",
                "exit_time",
                "age",
                "sex",
                "pregnancy",
            ]
        )

    def _get_mortality_probability(self, builder: Builder):
        probability_data = builder.data.load(
            data_keys.MATERNAL_DISORDERS.MORTALITY_PROBABILITY
        )
        probability_pipeline_source = builder.lookup.build_table(
            probability_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        return builder.value.register_value_producer(
            self.mortality_probability_pipeline_name,
            source=probability_pipeline_source,
            requires_columns=["age", "sex"],
        )

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[
                self.cause_of_death_column_name,
                self.years_of_life_lost_column_name,
            ],
        )

    def _register_on_timestep_listener(self, builder: Builder) -> None:
        builder.event.register_listener("time_step", self.on_time_step, priority=9)

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
            event.index, query="(alive == 'alive') & (pregnancy == 'parturition')"
        )
        mortality_probability = self.mortality_probability(pop.index)

        deaths = self.random.filter_for_probability(
            pop.index, mortality_probability, additional_key="death"
        )

        pop.loc[deaths, "alive"] = "dead"
        pop.loc[deaths, "exit_time"] = event.time
        pop.loc[deaths, "years_of_life_lost"] = self.life_expectancy(deaths)
        pop.loc[deaths, "cause_of_death"] = "maternal_disorders"
        self.population_view.update(pop)

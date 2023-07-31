import functools
import itertools
import time
from collections import Counter
from typing import Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium_public_health.metrics import DisabilityObserver as DisabilityObserver_
from vivarium_public_health.metrics import DiseaseObserver, MortalityObserver
from vivarium_public_health.metrics import ResultsStratifier as ResultsStratifier_
from vivarium_public_health.metrics.stratification import Source, SourceType
from vivarium_public_health.utilities import to_years

from vivarium_gates_nutrition_optimization.constants import data_values, models


class ResultsStratifier(ResultsStratifier_):
    def register_stratifications(self, builder: Builder) -> None:
        super().register_stratifications(builder)

        self.setup_stratification(
            builder,
            name="pregnancy_outcome",
            sources=[Source("pregnancy_outcome", SourceType.COLUMN)],
            categories=models.PREGNANCY_OUTCOMES,
        )

        self.setup_stratification(
            builder,
            name="pregnancy",
            sources=[Source("pregnancy", SourceType.COLUMN)],
            categories=models.PREGNANCY_MODEL_STATES,
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


class AnemiaObserver:
    configuration_defaults = {
        "observers": {
            "anemia": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __repr__(self):
        return "AnemiaObserver()"

    @property
    def name(self):
        return "anemia_observer"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.observers.anemia
        self.stratifier = builder.components.get_component(ResultsStratifier.name)

        self.person_time = Counter()

        self.anemia_levels = builder.value.get_value("anemia_levels")
        self.hemoglobin = builder.value.get_value("hemoglobin.exposure")

        columns_required = [
            "alive",
            "pregnancy",
        ]
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)
        builder.value.register_value_modifier("metrics", self.metrics)

    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        pop["anemia_level"] = self.anemia_levels(pop.index)
        step_size = to_years(event.step_size)

        anemia_levels = data_values.ANEMIA_DISABILITY_WEIGHTS.keys()

        new_person_time = {}
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
        for label, group_mask in groups:
            for anemia_level in anemia_levels:
                key = f"{anemia_level}_anemia_{label}"
                group = pop[group_mask & (pop["anemia_level"] == anemia_level)]
                new_person_time[key] = len(group) * step_size

        self.person_time.update(new_person_time)

    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        metrics.update(self.person_time)
        return metrics


class DisabilityObserver(DisabilityObserver_):
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.disability_pipelines["anemia"] = builder.value.get_value(
            "anemia.disability_weight"
        )

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


def timeit(name):
    def _wrapper(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(name, time.time() - start, " s")
            return result

        @functools.wraps(func)
        def _wrapped_passthrough(*args, **kwargs):
            return func(*args, **kwargs)

        return _wrapped_passthrough

    return _wrapper


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
        self.exposure = Counter()

        self.anemia_levels = builder.value.get_value("anemia_levels")
        self.hemoglobin = builder.value.get_value("hemoglobin.exposure")

        columns_required = [
            "alive",
            "pregnancy_status",
            "maternal_hemorrhage",
            "maternal_bmi_anemia_category",
        ]
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)
        builder.event.register_listener("collect_metrics", self.on_collect_metrics)
        builder.value.register_value_modifier("metrics", self.metrics)

    @timeit("anemia_tsp")
    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        pop["anemia_level"] = self.anemia_levels(pop.index)
        step_size = to_years(event.step_size)

        anemia_measures = list(
            itertools.product(
                data_values.ANEMIA_DISABILITY_WEIGHTS.keys(),
                models.PREGNANCY_MODEL_STATES,
                models.MATERNAL_HEMORRHAGE_STATES,
            )
        )
        anemia_masks = {}
        for anemia_level, pregnancy_status, hemorrhage_state in anemia_measures:
            key = (anemia_level, pregnancy_status, hemorrhage_state)
            mask = (
                (pop["anemia_level"] == anemia_level)
                & (pop["pregnancy_status"] == pregnancy_status)
                & (pop["maternal_hemorrhage"] == hemorrhage_state)
            )
            anemia_masks[key] = mask

        new_person_time = {}
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
        for label, group_mask in groups:
            for (
                anemia_level,
                pregnancy_status,
                hemorrhage_state,
            ), anemia_mask in anemia_masks.items():
                key = f"{anemia_level}_anemia_person_time_among_{pregnancy_status}_with_{hemorrhage_state}_{label}"
                group = pop[group_mask & anemia_mask]
                new_person_time[key] = len(group) * step_size

        self.person_time.update(new_person_time)

    @timeit("anemia_cm")
    def on_collect_metrics(self, event: Event):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        pop["hemoglobin"] = self.hemoglobin(event.index)

        pregnancy_measures = list(
            itertools.product(
                models.PREGNANCY_MODEL_STATES,
                models.MATERNAL_HEMORRHAGE_MODEL_STATES,
            )
        )

        anemia_masks = {}
        for pregnancy_status, hemorrhage_state in pregnancy_measures:
            key = (pregnancy_status, hemorrhage_state)
            mask = (pop["pregnancy_status"] == pregnancy_status) & (
                pop["maternal_hemorrhage"] == hemorrhage_state
            )
            anemia_masks[key] = mask

        new_exposures = {}
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
        for label, group_mask in groups:
            for (pregnancy_status, hemorrhage_state), pregnancy_mask in anemia_masks.items():
                key = f"hemoglobin_exposure_sum_among_{pregnancy_status}_with_{hemorrhage_state}_{label}"
                group = pop[group_mask & pregnancy_mask]
                new_exposures[key] = group.hemoglobin.sum()

        self.exposure.update(new_exposures)

    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        metrics.update(self.person_time)
        metrics.update(self.exposure)
        return metrics


class DisabilityObserver(DisabilityObserver_):
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.disability_pipelines["anemia"] = builder.value.get_value(
            "real_anemia.disability_weight"
        )

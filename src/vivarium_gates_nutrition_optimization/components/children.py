from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium_cluster_tools.utilities import mkdir

from vivarium_gates_nutrition_optimization.constants import (
    data_keys,
    data_values,
    models,
)


class NewChildren:
    def __init__(self):
        self.lbwsg = LBWSGDistribution()

    @property
    def name(self):
        return "child_status"

    @property
    def sub_components(self):
        return [self.lbwsg]

    @property
    def columns_created(self):
        return ["sex_of_child", "birth_weight"]

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.male_sex_percentage = data_values.INFANT_MALE_PERCENTAGES[
            builder.data.load(data_keys.POPULATION.LOCATION)
        ]

    def empty(self, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sex_of_child": models.INVALID_OUTCOME,
                "birth_weight": np.nan,
            },
            index=index,
        )

    def generate_children(self, index: pd.Index):
        sex_of_child = self.randomness.choice(
            index,
            choices=["Male", "Female"],
            p=[self.male_sex_percentage, 1 - self.male_sex_percentage],
            additional_key="sex_of_child",
        )
        lbwsg = self.lbwsg(sex_of_child)
        return pd.DataFrame(
            {
                "sex_of_child": sex_of_child,
                "birth_weight": lbwsg["birth_weight"],
                "gestational_age": lbwsg["gestational_age"],
            },
            index=index,
        )


class LBWSGDistribution:
    @property
    def name(self):
        return "lbwsg_distribution"

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.exposure = builder.data.load(data_keys.LBWSG.EXPOSURE).set_index("sex")
        self.category_intervals = self._get_category_intervals(builder)

    def __call__(self, newborn_sex: pd.Series):
        categorical_exposure = self._sample_categorical_exposure(newborn_sex)
        continuous_exposure = self._sample_continuous_exposure(categorical_exposure)
        return continuous_exposure

    ############
    # Sampling #
    ############

    def _sample_categorical_exposure(self, newborn_sex: pd.Series):
        categorical_exposures = []
        for sex in newborn_sex.unique():
            group_data = newborn_sex[newborn_sex == sex]
            sex_exposure = self.exposure.loc[sex]
            categorical_exposures.append(
                self.randomness.choice(
                    group_data.index,
                    choices=sex_exposure.parameter.tolist(),
                    p=sex_exposure.value.tolist(),
                    additional_key="categorical_exposure",
                )
            )
        categorical_exposures = pd.concat(categorical_exposures).sort_index()
        return categorical_exposures

    def _sample_continuous_exposure(self, categorical_exposure: pd.Series):
        intervals = self.category_intervals.loc[categorical_exposure]
        intervals.index = categorical_exposure.index
        exposures = []
        for axis in ["birth_weight", "gestational_age"]:
            draw = self.randomness.get_draw(categorical_exposure.index, additional_key=axis)
            lower, upper = intervals[f"{axis}_lower"], intervals[f"{axis}_upper"]
            exposures.append((lower + (upper - lower) * draw).rename(axis))
        return pd.concat(exposures, axis=1)

    ################
    # Data loading #
    ################

    def _get_category_intervals(self, builder: Builder):
        categories = builder.data.load(data_keys.LBWSG.CATEGORIES)
        category_intervals = pd.DataFrame(
            data=[
                (category, *self._parse_description(description))
                for category, description in categories.items()
            ],
            columns=[
                "category",
                "birth_weight_lower",
                "birth_weight_upper",
                "gestational_age_lower",
                "gestational_age_upper",
            ],
        ).set_index("category")
        return category_intervals

    @staticmethod
    def _parse_description(description: str) -> Tuple:
        birth_weight = [
            float(val) for val in description.split(", [")[1].split(")")[0].split(", ")
        ]
        gestational_age = [
            float(val) for val in description.split("- [")[1].split(")")[0].split(", ")
        ]
        return *birth_weight, *gestational_age


class BirthRecorder:
    @property
    def name(self):
        return "birth_recorder"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.output_path = self._build_output_path(builder)
        self.randomness = builder.randomness.get_stream(self.name)

        self.births = []

        required_columns = [
            "pregnancy_outcome",
            "pregnancy_duration",
            "birth_weight",
            "pregnancy",
            "previous_pregnancy",
        ]
        self.population_view = builder.population.get_view(required_columns)

        builder.event.register_listener("collect_metrics", self.on_collect_metrics)
        builder.event.register_listener("simulation_end", self.write_output)

    def on_collect_metrics(self, event: Event):
        pop = self.population_view.get(event.index)
        new_birth_mask = (
            (
                pop["pregnancy_outcome"].isin(
                    [models.LIVE_BIRTH_OUTCOME, models.STILLBIRTH_OUTCOME]
                )
            )
            & (pop["previous_pregnancy"] == models.PREGNANT_STATE_NAME)
            & (pop["pregnancy"] == models.POSTPARTUM_STATE_NAME)
        )

        new_births = pop.loc[new_birth_mask, ["pregnancy_duration", "birth_weight"]].rename(
            columns={"pregnancy_duration": "gestational_age"}
        )

        self.births.append(new_births)

    # noinspection PyUnusedLocal
    def write_output(self, event: Event) -> None:
        births_data = pd.concat(self.births)
        births_data.to_hdf(f"{self.output_path}.hdf", key="data")
        births_data.to_csv(f"{self.output_path}.csv")

    ###########
    # Helpers #
    ###########

    @staticmethod
    def _build_output_path(builder: Builder) -> Path:
        results_root = builder.configuration.output_data.results_directory
        output_root = Path(results_root) / "child_data"

        mkdir(output_root, exists_ok=True)

        input_draw = builder.configuration.input_data.input_draw_number
        seed = builder.configuration.randomness.random_seed
        output_path = output_root / f"draw_{input_draw}_seed_{seed}"

        return output_path

import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder

from vivarium_gates_nutrition_optimization.constants import (
    data_keys,
    data_values,
    models,
)


class NewChildren(Component):
    ##############
    # Properties #
    ##############

    @property
    def sub_components(self) -> List[str]:
        return [self.lbwsg]

    def __init__(self):
        super().__init__()
        self.lbwsg = LBWSGDistribution()

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
                "gestational_age": np.nan,
            },
            index=index,
        )

    def generate_children(self, index: pd.Index) -> pd.DataFrame:
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


class LBWSGDistribution(Component):
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
                (
                    category,
                    ga_interval.left,
                    ga_interval.right,
                    bw_interval.left,
                    bw_interval.right,
                )
                for category, description in categories.items()
                for ga_interval, bw_interval in [self._parse_description(description)]
            ],
            columns=[
                "category",
                "gestational_age_lower",
                "gestational_age_upper",
                "birth_weight_lower",
                "birth_weight_upper",
            ],
        ).set_index("category")
        return category_intervals

    @staticmethod
    def _parse_description(description: str) -> tuple[pd.Interval, pd.Interval]:
        """Parses a string corresponding to a low birth weight and short gestation
        category to an Interval.

        An example of a standard description:
        'Neonatal preterm and LBWSG (estimation years) - [0, 24) wks, [0, 500) g'
        An example of an edge case for gestational age:
        'Neonatal preterm and LBWSG (estimation years) - [40, 42+] wks, [2000, 2500) g'
        An example of an edge case of birth weight:
        'Neonatal preterm and LBWSG (estimation years) - [36, 37) wks, [4000, 9999] g'
        """
        lbwsg_values = [float(val) for val in re.findall(r"(\d+)", description)]
        if len(list(lbwsg_values)) != 4:
            raise ValueError(
                f"Could not parse LBWSG description '{description}'. Expected 4 numeric values."
            )
        return (
            pd.Interval(*lbwsg_values[:2], closed="left"),  # Gestational Age
            pd.Interval(*lbwsg_values[2:], closed="left"),  # Birth Weight
        )

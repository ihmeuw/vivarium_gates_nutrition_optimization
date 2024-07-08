from datetime import datetime
from functools import partial
from typing import Any, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer
from vivarium.framework.time import get_time_stamp
from vivarium_public_health.disease import DiseaseState
from vivarium_public_health.results import COLUMNS
from vivarium_public_health.results import DisabilityObserver as DisabilityObserver_
from vivarium_public_health.results import (
    DiseaseObserver,
    MortalityObserver,
    PublicHealthObserver,
)
from vivarium_public_health.results import ResultsStratifier as ResultsStratifier_
from vivarium_public_health.utilities import to_years

from vivarium_gates_nutrition_optimization.constants import data_values, models


class ResultsStratifier(ResultsStratifier_):
    def register_stratifications(self, builder: Builder) -> None:
        super().register_stratifications(builder)

        #         builder.results.register_stratification(
        #             "anemia_status_at_birth",
        #             data_values.ANEMIA_STATUS_AT_BIRTH_CATEGORIES,
        #             requires_columns=["anemia_status_at_birth"],
        #         )

        builder.results.register_stratification(
            "anemia_levels",
            data_values.ANEMIA_DISABILITY_WEIGHTS.keys(),
            requires_values=["anemia_levels"],
        )

        builder.results.register_stratification(
            "maternal_bmi_anemia_category",
            models.BMI_ANEMIA_CATEGORIES,
            requires_columns=["maternal_bmi_anemia_category"],
        )

        builder.results.register_stratification(
            "intervention",
            models.SUPPLEMENTATION_CATEGORIES,
            requires_columns=["intervention"],
        )

        builder.results.register_stratification(
            "pregnancy_outcome",
            models.PREGNANCY_OUTCOMES,
            requires_columns=["pregnancy_outcome"],
        )


class PregnancyObserver(DiseaseObserver):
    def __init__(self):
        super().__init__("pregnancy")


class MaternalMortalityObserver(MortalityObserver):
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        # Hack in maternal disorders
        maternal_disorders = DiseaseState(models.MATERNAL_DISORDERS_MODEL_NAME)
        maternal_disorders.set_model(models.MATERNAL_DISORDERS_MODEL_NAME)
        self.causes_of_death += [maternal_disorders]

    #     self.causes_to_stratify += [models.MATERNAL_DISORDERS_MODEL_NAME]

    # def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
    #     entity_type = super().get_entity_type_column(measure, results)
    #     # Fill in the missing values due to 'maternal_disorders' not being set
    #     # up as a propery DiseaseState
    #     return entity_type.fillna("cause")


class AnemiaObserver(PublicHealthObserver):
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": ["anemia_levels"],
                },
            },
        }

    def register_observations(self, builder: Builder) -> None:
        self.register_adding_observation(
            builder=builder,
            name=f"person_time_anemia",
            pop_filter=f'alive == "alive" and tracked == True',
            when="time_step__prepare",
            requires_columns=["alive"],
            requires_values=["anemia_levels"],
            additional_stratifications=builder.configuration.stratification.anemia.include,
            excluded_stratifications=builder.configuration.stratification.anemia.exclude,
            aggregator=partial(aggregate_state_person_time, builder.time.step_size()()),
        )

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        results = results.reset_index()
        results.rename(columns={"anemia_levels": "sub_entity"}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("person_time", index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("impairment", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("anemia", index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        # This column was created in the 'format' method
        return results[COLUMNS.SUB_ENTITY]


class MaternalBMIObserver(PublicHealthObserver):
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": ["maternal_bmi_anemia_category"],
                },
            },
        }

    def register_observations(self, builder: Builder) -> None:
        self.register_adding_observation(
            builder=builder,
            name=f"person_time_maternal_bmi_anemia",
            pop_filter=f'alive == "alive" and tracked == True',
            when="time_step__prepare",
            requires_columns=["alive", "maternal_bmi_anemia_category"],
            additional_stratifications=builder.configuration.stratification.maternal_bmi.include,
            excluded_stratifications=builder.configuration.stratification.maternal_bmi.exclude,
            aggregator=partial(aggregate_state_person_time, builder.time.step_size()()),
        )

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        results = results.reset_index()
        results.rename(columns={"maternal_bmi_anemia_category": "sub_entity"}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("person_time", index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("custom_risk_exposure", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("bmi_anemia", index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        # This column was created in the 'format' method
        return results[COLUMNS.SUB_ENTITY]


class MaternalInterventionObserver(PublicHealthObserver):
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": ["intervention"],
                },
            },
        }

    def register_observations(self, builder: Builder) -> None:
        # 2 weeks between administration and effect
        intervention_date = get_time_stamp(builder.configuration.time.start) + pd.Timedelta(
            days=data_values.DURATIONS.INTERVENTION_DELAY_DAYS - 2 * 7
        )
        self.register_adding_observation(
            builder=builder,
            name="intervention_count",
            pop_filter=(
                'alive == "alive" and tracked == True and '
                f'event_time > "{intervention_date}" and '
                f'event_time <= "{intervention_date + builder.time.step_size()()}"'
            ),
            requires_columns=["alive", "intervention", "event_time"],
            additional_stratifications=builder.configuration.stratification.maternal_intervention.include,
            excluded_stratifications=builder.configuration.stratification.maternal_intervention.exclude,
        )

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        results = results.reset_index()
        results.rename(columns={"intervention": "sub_entity"}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series(measure, index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("intervention", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("maternal_intervention", index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        # This column was created in the 'format' method
        return results[COLUMNS.SUB_ENTITY]


class PregnancyOutcomeObserver(PublicHealthObserver):
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": ["pregnancy_outcome"],
                },
            },
        }

    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.start_date = get_time_stamp(builder.configuration.time.start)

    def register_observations(self, builder: Builder) -> None:

        self.register_adding_observation(
            builder=builder,
            name=f"pregnancy_outcome_count",
            pop_filter="",
            requires_columns=["pregnancy_outcome"],
            additional_stratifications=builder.configuration.stratification.pregnancy_outcome.include,
            excluded_stratifications=builder.configuration.stratification.pregnancy_outcome.exclude,
            aggregator=self.count_pregnancy_outcomes_at_initialization,
        )

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        results = results.reset_index()
        results.rename(columns={"pregnancy_outcome": "sub_entity"}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series(measure, index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("custom_fertility", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("pregnancy_countcome", index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        # This column was created in the 'format' method
        return results[COLUMNS.SUB_ENTITY]

    ###############
    # Aggregators #
    ###############

    def count_pregnancy_outcomes_at_initialization(self, x: pd.DataFrame) -> float:
        if self.clock() == self.start_date:
            return len(x)
        else:
            return 0


class DisabilityObserver(DisabilityObserver_):
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        # Hack in Anemia
        anemia = DiseaseState("anemia")
        anemia.set_model("anemia")
        self.causes_of_disease += [anemia]


def aggregate_state_person_time(step_size, df: pd.DataFrame) -> float:
    return len(df) * to_years(step_size)


class BirthObserver(Observer):

    COL_MAPPING = {
        "sex_of_child": "sex",
        "birth_weight": "birth_weight",
        "maternal_bmi_anemia_category": "joint_bmi_anemia_category",
        "gestational_age": "gestational_age",
        "pregnancy_outcome": "pregnancy_outcome",
        "intervention": "maternal_intervention",
    }

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_concatenating_observation(
            name="births",
            pop_filter=(
                "("
                f"pregnancy_outcome == '{models.LIVE_BIRTH_OUTCOME}' "
                f"or pregnancy_outcome == '{models.STILLBIRTH_OUTCOME}'"
                ") "
                f"and previous_pregnancy == '{models.PREGNANT_STATE_NAME}' "
                f"and pregnancy == '{models.PARTURITION_STATE_NAME}'"
            ),
            requires_columns=list(self.COL_MAPPING),
            results_formatter=self.format,
        )

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        new_births = results[list(self.COL_MAPPING)].rename(columns=self.COL_MAPPING)
        new_births["birth_date"] = datetime(2024, 12, 30).strftime("%Y-%m-%d T%H:%M.%f")
        new_births["joint_bmi_anemia_category"] = new_births["joint_bmi_anemia_category"].map(
            {
                "low_bmi_anemic": "cat1",
                "normal_bmi_anemic": "cat2",
                "low_bmi_non_anemic": "cat3",
                "normal_bmi_non_anemic": "cat4",
            }
        )

        return new_births

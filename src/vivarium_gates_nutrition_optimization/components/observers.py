from functools import partial
from typing import Any, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer
from vivarium.framework.time import get_time_stamp
from vivarium_public_health.disease import DiseaseState
from vivarium_public_health.results import DisabilityObserver as DisabilityObserver_
from vivarium_public_health.results import DiseaseObserver, MortalityObserver
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
        self.causes_to_stratify += [models.MATERNAL_DISORDERS_MODEL_NAME]


class AnemiaObserver(Observer):
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
        builder.results.register_adding_observation(
            name=f"person_time_anemia",
            pop_filter=f'alive == "alive" and tracked == True',
            when="time_step__prepare",
            requires_columns=["alive"],
            requires_values=["anemia_levels"],
            additional_stratifications=builder.configuration.stratification.anemia.include,
            excluded_stratifications=builder.configuration.stratification.anemia.exclude,
            aggregator=partial(aggregate_state_person_time, builder.time.step_size()()),
        )


class MaternalBMIObserver(Observer):
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
        builder.results.register_adding_observation(
            name=f"person_time_maternal_bmi_anemia",
            pop_filter=f'alive == "alive" and tracked == True',
            when="time_step__prepare",
            requires_columns=["alive", "maternal_bmi_anemia_category"],
            additional_stratifications=builder.configuration.stratification.maternal_bmi.include,
            excluded_stratifications=builder.configuration.stratification.maternal_bmi.exclude,
            aggregator=partial(aggregate_state_person_time, builder.time.step_size()()),
        )


class MaternalInterventionObserver(Observer):
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
        builder.results.register_adding_observation(
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


class PregnancyOutcomeObserver(Observer):
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

        builder.results.register_adding_observation(
            name=f"pregnancy_outcome_count",
            pop_filter="",
            requires_columns=["pregnancy_outcome"],
            additional_stratifications=builder.configuration.stratification.pregnancy_outcome.include,
            excluded_stratifications=builder.configuration.stratification.pregnancy_outcome.exclude,
            aggregator=self.count_pregnancy_outcomes_at_initialization,
        )

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

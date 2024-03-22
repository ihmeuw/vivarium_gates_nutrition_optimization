from functools import partial

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import State
from vivarium.framework.time import get_time_stamp
from vivarium_public_health.metrics import DisabilityObserver as DisabilityObserver_
from vivarium_public_health.metrics import DiseaseObserver, MortalityObserver
from vivarium_public_health.metrics import ResultsStratifier as ResultsStratifier_
from vivarium_public_health.utilities import to_years

from vivarium_gates_nutrition_optimization.constants import data_values, models


class ResultsStratifier(ResultsStratifier_):
    def register_stratifications(self, builder: Builder) -> None:
        super().register_stratifications(builder)

        builder.results.register_stratification(
            "pregnancy_outcome",
            models.PREGNANCY_OUTCOMES,
            requires_columns=["pregnancy_outcome"],
        )

        builder.results.register_stratification(
            "pregnancy", models.PREGNANCY_MODEL_STATES, requires_columns=["pregnancy"]
        )

        builder.results.register_stratification(
            "anemia_status_at_birth",
            data_values.ANEMIA_STATUS_AT_BIRTH_CATEGORIES,
            requires_columns=["anemia_status_at_birth"],
        )

        builder.results.register_stratification(
            "intervention",
            models.SUPPLEMENTATION_CATEGORIES,
            requires_columns=["intervention"],
        )


class PregnancyObserver(DiseaseObserver):
    def __init__(self):
        super().__init__("pregnancy")


class MaternalMortalityObserver(MortalityObserver):
    def setup(self, builder: Builder):
        self.causes_of_death += [models.MATERNAL_DISORDERS_MODEL_NAME]
        super().setup(builder)


class AnemiaObserver(Component):
    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "anemia": {
                "exclude": [],
                "include": [],
            }
        }
    }

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification.anemia

        for anemia_category in data_values.ANEMIA_DISABILITY_WEIGHTS.keys():
            builder.results.register_observation(
                name=f"anemia_{anemia_category}_person_time",
                pop_filter=f'alive == "alive" and anemia_levels == "{anemia_category}" and tracked == True',
                aggregator=partial(aggregate_state_person_time, self.step_size()),
                requires_columns=["alive"],
                requires_values=["anemia_levels"],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
            )


class MaternalBMIObserver(Component):
    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "maternal_bmi": {
                "exclude": [],
                "include": [],
            }
        }
    }

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification.maternal_bmi

        for bmi_category in models.BMI_ANEMIA_CATEGORIES:
            builder.results.register_observation(
                name=f"maternal_bmi_anemia_{bmi_category}_person_time",
                pop_filter=f'alive == "alive" and maternal_bmi_anemia_category == "{bmi_category}" and tracked == True',
                aggregator=partial(aggregate_state_person_time, self.step_size()),
                requires_columns=["alive", "maternal_bmi_anemia_category"],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
            )


class MaternalInterventionObserver(Component):
    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "maternal_interventions": {
                "exclude": [],
                "include": [],
            }
        }
    }

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification.maternal_interventions
        intervention_date = get_time_stamp(builder.configuration.time.start) + pd.Timedelta(
            days=data_values.DURATIONS.INTERVENTION_DELAY_DAYS
            - 2 * 7
            ## 2 weeks between administration and effect
        )

        for intervention in models.SUPPLEMENTATION_CATEGORIES:
            builder.results.register_observation(
                name=f"intervention_{intervention}_count",
                pop_filter=f'alive == "alive" and intervention == "{intervention}" and tracked == True and event_time > "{intervention_date}" and event_time <= "{intervention_date + self.step_size()}"',
                requires_columns=["alive", "intervention"],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
            )


class PregnancyOutcomeObserver(Component):
    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "pregnancy_outcomes": {
                "exclude": [],
                "include": [],
            }
        }
    }

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification.pregnancy_outcomes

        for outcome in models.PREGNANCY_OUTCOMES:
            builder.results.register_observation(
                name=f"pregnancy_outcome_{outcome}_count",
                pop_filter=f'alive == "alive" and tracked == True'
                f'and previous_pregnancy == "pregnant" and pregnancy == "parturition"'
                f'and pregnancy_outcome == "{outcome}"',
                requires_columns=[
                    "alive",
                    "previous_pregnancy",
                    "pregnancy",
                    "pregnancy_outcome",
                ],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
            )


class DisabilityObserver(DisabilityObserver_):
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.stratification.disability
        self.step_size = pd.Timedelta(days=builder.configuration.time.step_size)
        self.disability_weight = self.get_disability_weight_pipeline(builder)
        cause_states = builder.components.get_components_by_type(
            tuple(self.disease_classes)
            # Hack in Anemia
        ) + [State("anemia")]
        base_query = 'tracked == True and alive == "alive"'

        builder.results.register_observation(
            name="ylds_due_to_all_causes",
            pop_filter=base_query,
            aggregator_sources=[self.disability_weight_pipeline_name],
            aggregator=self.disability_weight_aggregator,
            requires_columns=["alive"],
            requires_values=["disability_weight"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step__prepare",
        )

        for cause_state in cause_states:
            cause_disability_weight_pipeline_name = (
                f"{cause_state.state_id}.disability_weight"
            )
            builder.results.register_observation(
                name=f"ylds_due_to_{cause_state.state_id}",
                pop_filter=base_query
                if cause_state.state_id == "maternal_disorders"
                else base_query + ' and pregnancy != "parturition"',
                aggregator_sources=[cause_disability_weight_pipeline_name],
                aggregator=self.disability_weight_aggregator,
                requires_columns=["alive", "pregnancy"],
                requires_values=[cause_disability_weight_pipeline_name],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
            )


def aggregate_state_person_time(step_size, df: pd.DataFrame) -> float:
    return len(df) * to_years(step_size)

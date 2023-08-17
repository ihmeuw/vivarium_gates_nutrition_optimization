from functools import partial

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import State
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
        super().setup(builder)
        cause_of_death = models.MATERNAL_DISORDERS_MODEL_NAME
        self.causes_of_death += cause_of_death

        builder.results.register_observation(
            name=f"death_due_to_{cause_of_death}",
            pop_filter=f'alive == "dead" and cause_of_death == "{cause_of_death}" and tracked == True',
            aggregator=self.count_cause_specific_deaths,
            requires_columns=["alive", "cause_of_death", "exit_time"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
        )
        builder.results.register_observation(
            name=f"ylls_due_to_{cause_of_death}",
            pop_filter=f'alive == "dead" and cause_of_death == "{cause_of_death}" and tracked == True',
            aggregator=self.calculate_cause_specific_ylls,
            requires_columns=[
                "alive",
                "cause_of_death",
                "exit_time",
                "years_of_life_lost",
            ],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
        )


class AnemiaObserver:
    configuration_defaults = {
        "stratification": {
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


class MaternalBMIObserver:
    configuration_defaults = {
        "stratification": {
            "maternal_bmi": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __repr__(self):
        return "MaternalBMIObserver()"

    @property
    def name(self):
        return "maternal_bmi_observer"

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


class MaternalInterventionObserver:
    configuration_defaults = {
        "stratification": {
            "maternal_interventions": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __repr__(self):
        return "MaternalInterventionObserver()"

    @property
    def name(self):
        return "maternal_intervention_observer"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification.maternal_interventions

        for intervention in models.SUPPLEMENTATION_CATEGORIES:
            builder.results.register_observation(
                name=f"intervention_{intervention}_count",
                pop_filter=f'alive == "alive" and intervention == "{intervention}" and tracked == True',
                requires_columns=["alive", "intervention"],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
            )

class PregnancyOutcomeObserver:
    configuration_defaults = {
        "stratification": {
            "pregnancy_outcomes": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __repr__(self):
        return "PregnancyOutcomeObserver()"

    @property
    def name(self):
        return "pregnancy_outcome_observer"

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
                pop_filter=f'alive == "alive" and pregnancy_outcome == "{outcome}" and tracked == True',
                requires_columns=["alive", "pregnancy_outcome"],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
            )

class DisabilityObserver(DisabilityObserver_):
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.stratification.disability
        self.step_size = pd.Timedelta(days=builder.configuration.time.step_size)
        self.disability_weight = self.get_disability_weight_pipeline(builder)
        #Hack in Anemia
        cause_states = builder.components.get_components_by_type(tuple(self.disease_classes)) + [State("anemia")]
        base_query = 'tracked == True and alive == "alive"'

        builder.results.register_observation(
            name="ylds_due_to_all_causes",
            pop_filter=base_query,
            aggregator_sources=[self.disability_weight_pipeline_name],
            aggregator=self._disability_weight_aggregator,
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
                pop_filter=base_query if cause_state.state_id == 'maternal_disorders' else base_query + ' and pregnancy != "parturition"',
                aggregator_sources=[cause_disability_weight_pipeline_name],
                aggregator=self._disability_weight_aggregator,
                requires_columns=["alive"],
                requires_values=[cause_disability_weight_pipeline_name],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
            )


def aggregate_state_person_time(step_size, df: pd.DataFrame) -> float:
    return len(df) * to_years(step_size)

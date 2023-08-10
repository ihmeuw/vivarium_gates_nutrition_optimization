import pandas as pd
from vivarium.framework.engine import Builder
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
                aggregator=lambda x:  len(x) * to_years(self.step_size()),
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
                name=f"maternal_bmi_anemia_category_{bmi_category}_person_time",
                pop_filter=f'alive == "alive" and maternal_bmi_anemia_category == "{bmi_category}" and tracked == True',
                aggregator=lambda x:  len(x) * to_years(self.step_size()),
                requires_columns=["alive","maternal_bmi_anemia_category"],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
            )

class DisabilityObserver(DisabilityObserver_):
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        builder.results.register_observation(
            name=f"ylds_due_to_anemia",
            pop_filter='tracked == True and alive == "alive"',
            aggregator_sources=["anemia.disability_weight"],
            aggregator=self._disability_weight_aggregator,
            requires_columns=["alive"],
            requires_values=["anemia.disability_weight"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step__prepare",
        )